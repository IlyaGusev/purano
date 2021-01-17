import argparse
import csv

from purano.io import read_markup_tsv


def convert(
    input_file,
    output_file,
    answers_only,
    data_only
):
    assert answers_only and not data_only or data_only and not answers_only

    records = read_markup_tsv(input_file)
    fields = tuple()
    if answers_only:
        fields = ("first_url", "second_url", "quality")
    elif data_only:
        fields = (
            "first_url",
            "second_url",
            "first_title",
            "second_title",
            "first_text",
            "second_text"
        )

    with open(output_file, "w") as w:
        writer = csv.writer(w, delimiter='\t', quotechar='"')
        for r in records:
            writer.writerow([r[field] for field in fields])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--answers-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    convert(**vars(args))
