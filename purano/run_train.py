import json

from fasttext import load_model as ft_load_model
from pyonmttok import Tokenizer
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from purano.training.datasets import Text2TitleDataset
from purano.training.models import Text2TitleModel

MAX_WORDS = 150
BATCH_SIZE = 64
NUM_WORKERS = 5
EPOCHS = 10

train_start = 200000
train_end = 300000
train_step = 2

val_start = 400000
val_end = 420000
val_step = 2

ft_model = ft_load_model("models/fasttext/ru_vectors_v3.bin")
tokenizer = Tokenizer("conservative", joiner_annotate=False)

print("Fetch data")
tg_data = []
with open("data/documents/ru_tg_1101_0510.jsonl", "r") as r:
    for line in r:
        tg_data.append(json.loads(line))
tg_data.sort(key=lambda x: x['timestamp'])

print("Build datasets")
train_data = Text2TitleDataset(tg_data[train_start:train_end:train_step], ft_model, tokenizer, max_words=MAX_WORDS)
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

val_data = Text2TitleDataset(tg_data[val_start:val_end:val_step], ft_model, tokenizer, max_words=MAX_WORDS)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

print("Model training")
model = Text2TitleModel()
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0,
    patience=4,
    verbose=True,
    mode="min"
)
trainer = Trainer(
    gpus=0,
    checkpoint_callback=False,
    accumulate_grad_batches=1,
    max_epochs=EPOCHS,
    callbacks=[early_stop_callback],
    val_check_interval=0.5,
    progress_bar_refresh_rate=100)
trainer.fit(model, train_loader, val_loader)
model.save("models/title.pt", "models/text.pt")
