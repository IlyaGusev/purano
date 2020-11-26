import numpy as np
from hnswlib import Index as HnswIndex

from purano.util import tokenize

class FasttextHnsw:
    def __init__(self, model):
        self.model = model
        self.vector_dim = model.get_dimension()
        self.hnsw = HnswIndex(space='l2', dim=self.vector_dim * 3)

    def words_to_embeddings(self, words):
        embeddings = np.zeros((len(words), self.vector_dim))
        for i, w in enumerate(words):
            embeddings[i] = self.model.get_word_vector(w)
            embeddings[i] /= np.linalg.norm(embeddings[i])
        return embeddings

    def embed_text(self, text):
        words = tokenize(text)
        norm_vectors = self.words_to_embeddings(words)
        avg_wv = np.mean(norm_vectors, axis=0)
        max_wv = np.max(norm_vectors, axis=0)
        min_wv = np.min(norm_vectors, axis=0)
        return np.concatenate((avg_wv, max_wv, min_wv))

    def build_hnsw(self, texts):
        n = len(texts)
        self.hnsw.init_index(max_elements=n, ef_construction=100, M=16)
        embeddings = np.zeros((n, self.vector_dim * 3))
        for i, text in enumerate(texts):
            embeddings[i] = self.embed_text(text)
        self.hnsw.add_items(embeddings)
