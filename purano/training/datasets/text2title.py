import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from purano.util import tokenize
from purano.training.fasttext_hnsw import FasttextHnsw

class Text2TitleDataset(Dataset):
    def __init__(self, data, ft_model, min_words=2, max_words=150):
        self.ft_hnsw = FasttextHnsw(ft_model)
        self.ft_hnsw.build_hnsw([r["title"] for r in data])

        self.samples = []
        for count, row in enumerate(data):
            if count % 10000 == 0:
                print(count)
            title_words = tokenize(row["title"])
            text_words = tokenize(row["text"])[:max_words]
            if len(text_words) < min_words or len(title_words) < min_words:
                continue
            title = " ".join(title_words)
            text = " ".join(text_words)

            text_vector = self.ft_hnsw.embed_text(text)
            title_vector = self.ft_hnsw.embed_text(title)

            labels = list(self.ft_hnsw.hnsw.knn_query(title_vector, k=20)[0][0])[1:]
            random.shuffle(labels)
            bad_indices = labels[:2] + [random.randint(0, len(data) - 1)]
            bad_vectors = self.ft_hnsw.hnsw.get_items(bad_indices)
            for bad_vector in bad_vectors:
                bad_vector = np.array(bad_vector)
                assert bad_vector.shape == title_vector.shape == text_vector.shape
                assert bad_vector.dtype == title_vector.dtype == text_vector.dtype
                sample = (
                    torch.FloatTensor(text_vector),
                    torch.FloatTensor(title_vector),
                    torch.FloatTensor(bad_vector)
                )
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
