import argparse
import copy
import json
import os
import random

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from fasttext import load_model as ft_load_model
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers.neptune import NeptuneLogger

from purano.io import read_tg_jsonl
from purano.training.datasets import Text2TitleDataset
from purano.training.models import Text2TitleModel
from purano.util import get_true_file


def train_text2title(
    config_file: str,
    train_file: str,
    val_file: str,
    train_sample_rate: float,
    val_sample_rate: float,
    output_title_model_path: str,
    output_text_model_path: str,
    random_seed: int,
    neptune_project: str
):
    seed_everything(random_seed)

    train_file = get_true_file(train_file)
    val_file = get_true_file(val_file)
    assert train_file.endswith(".jsonl")
    assert val_file.endswith(".jsonl")

    config = json.loads(jsonnet_evaluate_file(config_file))

    print("Loading vectors...")
    ft_model_path = config.pop("ft_vector_model_path", "models/fasttext/ru_vectors_v3.bin")
    ft_model = ft_load_model(ft_model_path)

    print("Fetching data...")
    train_records = [r for r in read_tg_jsonl(train_file) if random.random() <= train_sample_rate]
    val_records = [r for r in read_tg_jsonl(val_file) if random.random() <= val_sample_rate]

    print("Building datasets...")
    max_words = config.get("max_words", 150)
    batch_size = config.get("batch_size", 64)
    num_workers = config.get("num_workers", 5)
    train_data = Text2TitleDataset(train_records, ft_model, max_words=max_words)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers
    )

    val_data = Text2TitleDataset(val_records, ft_model, max_words=max_words)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)

    print("Training model...")
    epochs = config.get("epochs", 100)
    patience = config.get("patience", 4)
    model = Text2TitleModel()
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=patience,
        verbose=True,
        mode="min"
    )
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
            experiment_name="Fasttext text2title",
            tags=["training", "pytorch-lightning", "text2title"],
            params=params
        )
    trainer = Trainer(
        gpus=0,
        checkpoint_callback=False,
        accumulate_grad_batches=1,
        max_epochs=epochs,
        callbacks=[early_stop_callback],
        val_check_interval=1.0,
        progress_bar_refresh_rate=100,
        deterministic=True,
        logger=logger
    )
    trainer.fit(model, train_loader, val_loader)
    model.save(output_title_model_path, output_text_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--train-sample-rate", type=float, default=1.0)
    parser.add_argument("--val-sample-rate", type=float, default=1.0)
    parser.add_argument("--output-title-model-path", type=str,
                        default="models/text2title/ru_ft_title_embedder.pt")
    parser.add_argument("--output-text-model-path", type=str,
                        default="models/text2title/ru_ft_text_embedder.pt")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--neptune-project", type=str, default=None)

    args = parser.parse_args()
    train_text2title(**vars(args))
