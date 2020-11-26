import argparse
import json
import random

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from transformers import AutoTokenizer, EncoderDecoderModel, Trainer, TrainingArguments, logging

from purano.readers.tg_jsonl import parse_tg_jsonl
from purano.training.datasets import GenTitleDataset
from purano.util import get_true_file


def train_gen_title(
    config_file: str,
    train_file: str,
    val_file: str,
    train_sample_rate: float,
    val_sample_rate: float,
    output_model_path: str,
):
    train_file = get_true_file(train_file)
    val_file = get_true_file(val_file)
    assert train_file.endswith(".jsonl")
    assert val_file.endswith(".jsonl")
    logging.set_verbosity_info()

    config = json.loads(jsonnet_evaluate_file(config_file))

    print("Fetching data...")
    train_records = [r for r in parse_tg_jsonl(train_file) if random.random() <= train_sample_rate]
    val_records = [r for r in parse_tg_jsonl(val_file) if random.random() <= val_sample_rate]

    print("Building datasets...")
    model_path = config.pop("model_path")
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)

    max_tokens_text = config.pop("max_tokens_text", 196)
    max_tokens_title = config.pop("max_tokens_title", 48)
    bos_token_id = config.pop("bos_token_id", tokenizer.bos_token_id)
    eos_token_id = config.pop("eos_token_id", tokenizer.eos_token_id)

    train_dataset = GenTitleDataset(
        train_records,
        tokenizer,
        max_tokens_text=max_tokens_text,
        max_tokens_title=max_tokens_title)

    val_dataset = GenTitleDataset(
        val_records,
        tokenizer,
        max_tokens_text=max_tokens_text,
        max_tokens_title=max_tokens_title)

    print("Initializing model...")
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_path, model_path)

    print("Training model...")
    batch_size = config.pop("batch_size", 8)
    eval_steps = config.pop("eval_steps", 10000)
    save_steps = config.pop("save_steps", 10000)
    logging_steps = config.pop("logging_steps", 100)
    learning_rate = config.pop("learning_rate", 5e-05)
    warmup_steps = config.pop("warmup_steps", 2000)
    num_train_epochs = config.pop("num_train_epochs", 5)
    training_args = TrainingArguments(
        output_dir=output_model_path,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluate_during_training=True,
        do_train=True,
        do_eval=True,
        overwrite_output_dir=False,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_total_limit=1,
        num_train_epochs=num_train_epochs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    model.save_pretrained(output_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--train-sample-rate", type=float, default=1.0)
    parser.add_argument("--val-sample-rate", type=float, default=1.0)
    parser.add_argument("--output-model-path", type=str, default="models/gen_title")

    args = parser.parse_args()
    train_gen_title(**vars(args))
