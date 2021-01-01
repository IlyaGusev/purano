import argparse
import json

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import torch
from tqdm import tqdm

from purano.readers.tg_jsonl import parse_tg_jsonl
from purano.training.models.tfidf import build_idf_vocabulary, get_tfidf_vector, SVDEmbedder
from purano.util import get_true_file


def train_tfidf(
    config_file,
    input_file,
    output_file,
    svd_matrix_file
):
    config = json.loads(jsonnet_evaluate_file(config_file))
    input_file = get_true_file(input_file)
    assert input_file.endswith(".jsonl")

    print("Parsing input data...")
    corpus = []
    for record in tqdm(parse_tg_jsonl(input_file)):
        corpus.append(record.pop("title") + " " + record.pop("text"))

    idfs = build_idf_vocabulary(corpus, **config.pop("building"))

    print("Saving vocabulary with IDFs...")
    with open(output_file, "w") as w:
        for word, idf in idfs:
            w.write("{}\t{}\n".format(word, idf))

    word2idf = {word: idf for word, idf in idfs}
    word2idx = {word: idx for idx, (word, _) in enumerate(idfs)}

    print("Preparing CSR martix...")
    X_data = []
    X_col_ind = []
    X_row_ind = []
    for i, text in enumerate(corpus):
        data, col_ind = get_tfidf_vector(text, word2idf, word2idx)
        row_ind = [i for _ in range(len(col_ind))]
        X_data += data
        X_col_ind += col_ind
        X_row_ind += row_ind
    X = csr_matrix((X_data, (X_row_ind, X_col_ind)))

    print("Calculating truncated SVD...")
    svd_dim = config.pop("svd_dim")
    svd = TruncatedSVD(n_components=svd_dim, n_iter=100, random_state=42)
    svd.fit(X)
    matrix = svd.components_.T
    model = SVDEmbedder(len(word2idf), svd_dim)
    model.mapping_layer.weight.data = torch.DoubleTensor(matrix).transpose(0, 1)
    torch.save(model, svd_matrix_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--svd-matrix-file", type=str, required=True)

    args = parser.parse_args()
    train_tfidf(**vars(args))
