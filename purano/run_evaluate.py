import argparse
import csv
import json
import os

from purano.clusterer.metrics import parse_threads_json, calc_metrics
from purano.readers import parse_tg_jsonl, parse_clustering_markup_tsv


def evaluate(
    clustering_markup_tsv,
    original_jsonl,
    threads_json,
    output_json,
    include_errors
):
    assert os.path.isfile(clustering_markup_tsv)
    assert clustering_markup_tsv.endswith(".tsv")
    assert os.path.isfile(original_jsonl)
    assert original_jsonl.endswith(".jsonl")
    assert os.path.isfile(threads_json)
    assert threads_json.endswith(".json")
    assert output_json.endswith(".json")

    markup = parse_clustering_markup_tsv(clustering_markup_tsv)
    url2record = {r["url"]: r for r in parse_tg_jsonl(original_jsonl)}
    labels = parse_threads_json(threads_json)
    metrics, errors = calc_metrics(markup, url2record, labels)

    with open(output_json, "w") as w:
        output = {"threads_metrics": metrics}
        if include_errors:
            output["threads_errors"] = errors
        json.dump(output, w, ensure_ascii=False, indent=4)

    for error in errors:
        print(error["target"], error["prediction"], " ||| ", error["first_title"], " ||| ", error["second_title"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clustering-markup-tsv", type=str, required=True)
    parser.add_argument("--original-jsonl", type=str, required=True)
    parser.add_argument("--threads-json", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--include-errors", default=False,  action='store_true')
    args = parser.parse_args()
    evaluate(**vars(args))
