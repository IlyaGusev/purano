from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn

from purano.util import tokenize_to_lemmas


def build_idf_vocabulary(texts, max_df=0.2, min_df=20):
    print("Building TfidfVectorizer...")
    vectorizer = TfidfVectorizer(tokenizer=tokenize_to_lemmas, max_df=max_df, min_df=min_df)
    vectorizer.fit(texts)
    idf_vector = vectorizer.idf_.tolist()

    print("{} words in vocabulary".format(len(idf_vector)))
    idfs = list()
    for word, idx in vectorizer.vocabulary_.items():
        idfs.append((word, idf_vector[idx]))

    idfs.sort(key=lambda x: (x[1], x[0]))
    return idfs


def load_idfs(file_name):
    word2idf = dict()
    word2idx = dict()
    with open(file_name, "r") as r:
        idx = 0
        for line in r:
            word, idf = line.strip().split("\t")
            word2idf[word] = float(idf)
            word2idx[word] = idx
            idx += 1
    return word2idf, word2idx


def get_tfidf_vector(text, word2idf, word2idx):
    tokens = tokenize_to_lemmas(text)
    freqs = Counter(tokens)
    tfs = {token: float(f) / len(tokens) for token, f in freqs.items()}
    data = []
    indices = []
    for token, tf in tfs.items():
        idx = word2idx.get(token, None)
        if idx is None:
            continue
        idf = word2idf.get(token, 0.0)
        tfidf = tf * idf
        data.append(tfidf)
        indices.append(idx)
    return data, indices


class SVDEmbedder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.mapping_layer = nn.Linear(embedding_dim, hidden_dim, bias=False)

    def forward(self, in_vectors):
        return self.mapping_layer(in_vectors.double())
