import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class Embedder(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=50):
        super().__init__()
        self.mapping_layer = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, in_vectors):
        projections = self.mapping_layer(in_vectors)
        norm = projections.norm(p=2, dim=1, keepdim=True)
        projections = projections.div(norm)
        return projections


class Text2TitleModel(LightningModule):
    def __init__(self, embedding_dim=384, hidden_dim=128):
        super().__init__()

        self.text_embedder = Embedder(embedding_dim, hidden_dim)
        self.title_embedder = Embedder(embedding_dim, hidden_dim)
        self.distance = nn.PairwiseDistance(p=2)
        self.margin = 0.3

    def save(self, title_embedder_filename, text_embedder_filename):
        torch.save(self.title_embedder, title_embedder_filename)
        torch.save(self.text_embedder, text_embedder_filename)

    def forward(self, pivot_vectors, positive_vectors, negative_vectors):
        pivot = self.text_embedder(pivot_vectors)
        positive = self.title_embedder(positive_vectors)
        negative = self.title_embedder(negative_vectors)
        distances = self.distance(pivot, positive) - self.distance(pivot, negative) + self.margin
        loss = torch.mean(torch.max(distances, torch.zeros_like(distances)))
        return loss

    def training_step(self, batch, batch_nb):
        train_loss = self(*batch)
        self.log("train_loss", train_loss, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_nb):
        val_loss = self(*batch)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_nb):
        test_loss = self(*batch)
        self.log("test_loss", test_loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return [optimizer]
