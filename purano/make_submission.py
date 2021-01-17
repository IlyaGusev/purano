import argparse
import csv
import os

from purano.io import read_markup_tsv, read_threads_json


def make_submission(
    clustering_markup_tsv,
    threads_json,
    output_file
):
    assert os.path.isfile(clustering_markup_tsv)
    assert clustering_markup_tsv.endswith(".tsv")
    assert os.path.isfile(threads_json)
    assert threads_json.endswith(".json")

    markup = read_markup_tsv(clustering_markup_tsv)
    labels = read_threads_json(threads_json)

    with open(output_file, "w") as w:
        writer = csv.writer(w, delimiter='\t', quotechar='"')
        for record in markup:
            first_url = record["first_url"]
            second_url = record["second_url"]
            if first_url not in labels:
                print("Missing url: {}".format(first_url))
                is_ok = False
            elif second_url not in labels:
                print("Missing url: {}".format(second_url))
                is_ok = False
            else:
                is_ok = labels[first_url] == labels[second_url]
            writer.writerow((first_url, second_url, "OK" if is_ok else "BAD"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clustering-markup-tsv", type=str, required=True)
    parser.add_argument("--threads-json", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()
    make_submission(**vars(args))
