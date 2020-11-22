import argparse
import json

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from purano.readers.tg_jsonl import parse_tg_jsonl
from purano.training.models.tfidf import build_idf_vocabulary


def train_tfidf(
    config_file,
    input_file,
    output_file
):
    assert input_file.endswith(".jsonl")
    config = json.loads(jsonnet_evaluate_file(config_file))

    print("Parsing input data...")
    corpus = []
    for record in parse_tg_jsonl(input_file):
        corpus.append(record["text"])
        corpus.append(record["title"])

    idfs = build_idf_vocabulary(corpus, **config)

    print("Saveing vocabulary with IDFs...")
    with open(output_file, "w") as w:
        for word, idf in idfs:
            w.write("{}\t{}\n".format(word, idf))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)

    args = parser.parse_args()
    train_tfidf(**vars(args))
