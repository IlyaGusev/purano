import json
import random

import numpy as np
import torch

from purano.training.datasets.fasttext_based import FastTextBasedDataset
from purano.util import tokenize

class Text2TitleDataset(FastTextBasedDataset):
    def __init__(self, data, ft_model, tokenizer, min_words=2, max_words=150):
        super().__init__(ft_model, tokenizer)
        self.hnsw = self.build_titles_hnsw(data)

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

            text_vector = self.embed_text(text)
            title_vector = self.embed_text(title)

            labels = list(self.hnsw.knn_query(title_vector, k=20)[0][0])[1:]
            random.shuffle(labels)
            bad_indices = labels[:2] + [random.randint(0, len(data) - 1)]
            bad_vectors = self.hnsw.get_items(bad_indices)
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
