from typing import Dict

import torch
from torch import nn
from transformers import DistilBertModel, DistilBertConfig, AdamW, get_linear_schedule_with_warmup
from pytorch_lightning import LightningModule


class DistilEmbeddingBertLightning(LightningModule):
    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        self.model_config = DistilBertConfig(**self.config["model"])
        self.model = DistilBertModel(self.model_config)
        self.criterion = nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')

    def forward(self, input_ids, attention_mask, embedding):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        target_embedding = embedding
        predicted_embedding = output.hidden_states[-1][:, 0, :]
        batch_size = target_embedding.size(0)
        loss = self.criterion(target_embedding, predicted_embedding, torch.ones(batch_size))
        return loss

    def training_step(self, batch, batch_nb):
        train_loss = self(**batch)
        self.log("train_loss", train_loss, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_nb):
        val_loss = self(**batch)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config["num_warmup_steps"],
            num_training_steps=self.config["num_training_steps"]
        )
        return [optimizer], [scheduler]
