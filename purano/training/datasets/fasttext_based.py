import numpy as np

from hnswlib import Index as HnswIndex
from torch.utils.data import Dataset

class FastTextBasedDataset(Dataset):
    def __init__(self, ft_model, tokenizer):
        self.ft_model = ft_model
        self.tokenizer = tokenizer

    def preprocess(self, text):
        text = str(text).strip().replace("\n", " ").replace("\xa0", " ").lower()
        tokens, _ = self.tokenizer.tokenize(text)
        text = " ".join(tokens)
        return text

    def words_to_embeddings(self, words):
        vector_dim = self.ft_model.get_dimension()
        embeddings = np.zeros((len(words), vector_dim))
        for i, w in enumerate(words):
            embeddings[i] = self.ft_model.get_word_vector(w)
            embeddings[i] /= np.linalg.norm(embeddings[i])
        return embeddings

    def embed_text(self, text):
        words = text.split(" ")
        norm_vectors = self.words_to_embeddings(words)
        avg_wv = np.mean(norm_vectors, axis=0)
        max_wv = np.max(norm_vectors, axis=0)
        min_wv = np.min(norm_vectors, axis=0)
        return np.concatenate((avg_wv, max_wv, min_wv))

    def build_titles_hnsw(self, data):
        vector_dim = self.ft_model.get_dimension() * 3
        hnsw = HnswIndex(space='l2', dim=vector_dim)
        hnsw.init_index(max_elements=len(data), ef_construction=100, M=16)
        embeddings = np.zeros((len(data), vector_dim))
        for i, record in enumerate(data):
            title = self.preprocess(record["title"])
            embeddings[i] = self.embed_text(title)
        hnsw.add_items(embeddings)
        return hnsw
