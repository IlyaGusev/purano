import argparse
import json
import itertools
import os
import random
from typing import List

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers.neptune import NeptuneLogger
from transformers import AutoModel, AutoTokenizer

from purano.readers.tg_jsonl import parse_tg_jsonl
from purano.util import get_true_file
from purano.training.datasets.embeddings_as_target import EmbeddingsAsTargetDataset
from purano.training.models.distil_bert import DistilEmbeddingBertLightning


def calc_batch_embeddings(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    max_tokens_count: int
):
    inputs = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_tokens_count,
        padding="max_length",
        truncation=True,
        return_tensors='pt'
    )
    batch_input_ids = inputs["input_ids"]
    batch_mask = inputs["attention_mask"]
    output = model(
        input_ids=batch_input_ids,
        attention_mask=batch_mask,
        return_dict=True,
        output_hidden_states=True
    )
    return output.hidden_states[-1].cpu().detach().numpy()[:, 0, :]


def distil_embeddings(
    config_file: str,
    train_file: str,
    val_file: str,
    train_sample_rate: float,
    val_sample_rate: float,
    input_model_path: str,
    output_model_path: str,
    random_seed: int,
    neptune_project: str,
    saved_embeddings: str
):
    seed_everything(random_seed)

    train_file = get_true_file(train_file)
    val_file = get_true_file(val_file)
    assert train_file.endswith(".jsonl")
    assert val_file.endswith(".jsonl")

    config = json.loads(jsonnet_evaluate_file(config_file))

    print("Fetching data...")
    train_records = [r for r in parse_tg_jsonl(train_file) if random.random() <= train_sample_rate]
    val_records = [r for r in parse_tg_jsonl(val_file) if random.random() <= val_sample_rate]

    tokenizer = AutoTokenizer.from_pretrained(input_model_path)
    max_tokens_count = config.get("max_tokens_count", 196)
    if not saved_embeddings or not os.path.isfile(saved_embeddings):
        print("Loading teacher model...")
        input_model = AutoModel.from_pretrained(input_model_path)

        print("Saving embeddings...")
        url2text = {r["url"]: r["text"] for r in itertools.chain(train_records, val_records)}
        urls = []
        embeddings = []
        batch_urls = []
        batch_texts = []
        batch_size = 8
        for url, text in tqdm(url2text.items()):
            batch_urls.append(url)
            batch_texts.append(text)
            if len(batch_urls) == batch_size:
                urls.extend(batch_urls)
                batch_embeddings = calc_batch_embeddings(
                    batch_texts,
                    tokenizer,
                    input_model,
                    max_tokens_count
                )
                for embedding in batch_embeddings:
                    embeddings.append(embedding)
                batch_urls = []
                batch_texts = []
        if batch_urls:
            urls.extend(batch_urls)
            batch_embeddings = calc_batch_embeddings(
                batch_texts,
                tokenizer,
                input_model,
                max_tokens_count
            )
            for embedding in batch_embeddings:
                embeddings.append(embedding)
        embeddings = torch.tensor(embeddings)
        data = {
            "urls": urls,
            "embeddings": embeddings
        }
        torch.save(data, saved_embeddings)
    else:
        print("Loading embeddings...")
        data = torch.load(saved_embeddings)

    url2num = {url: num for num, url in enumerate(data["urls"])}
    num2embedding = data["embeddings"]

    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 5)
    train_dataset = EmbeddingsAsTargetDataset(
        train_records,
        url2num,
        num2embedding,
        tokenizer,
        max_tokens_count
    )
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers
    )

    val_dataset = EmbeddingsAsTargetDataset(
        val_records,
        url2num,
        num2embedding,
        tokenizer,
        max_tokens_count
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    patience = config.get("patience", 4)
    epochs = config.get("epochs", 5)
    gradient_clip_val = config.get("gradient_clip_val", 1.0)

    logger = False
    neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")
    if neptune_project and neptune_api_token:
        params = copy.copy(config)
        params["train_sample_rate"] = train_sample_rate
        params["val_sample_rate"] = val_sample_rate
        params["train_file"] = train_file
        params["val_file"] = val_file
        logger = NeptuneLogger(
            api_key=neptune_api_token,
            project_name=neptune_project,
            experiment_name="Distil embeddings",
            tags=["training", "pytorch-lightning", "distil"],
            params=params
        )

    lightning_model = DistilEmbeddingBertLightning(config)
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=patience,
        verbose=True,
        mode="min"
    )
    trainer = Trainer(
        gpus=0,
        checkpoint_callback=False,
        accumulate_grad_batches=1,
        max_epochs=epochs,
        callbacks=[early_stop_callback],
        val_check_interval=1.0,
        gradient_clip_val=gradient_clip_val,
        deterministic=True,
        logger=logger
    )
    trainer.fit(lightning_model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--train-sample-rate", type=float, default=1.0)
    parser.add_argument("--val-sample-rate", type=float, default=1.0)
    parser.add_argument("--input-model-path", type=str, required=True)
    parser.add_argument("--output-model-path", type=str, required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--neptune-project", type=str, default=None)
    parser.add_argument("--saved-embeddings", type=str, default=None)

    args = parser.parse_args()
    distil_embeddings(**vars(args))
