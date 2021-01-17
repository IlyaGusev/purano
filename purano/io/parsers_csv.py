import csv
from urllib.parse import urlsplit
from dateutil.parser import parse as parse_datetime

from purano.io.parse_dir import parse_dir


def read_parsers_csv(file_name):
    with open(file_name, "r", encoding="utf-8") as r:
        header = next(r).strip().split(",")
        for line in r:
            line = line.replace("\\n", "\\\\n").replace("\\r", " ").replace("\t", " ")
            row = list(csv.reader([line], delimiter=",", quotechar='\"', escapechar='\\'))[0]
            record = dict(zip(header, row))
            if not record["text"] or not record["title"] or not record["url"]:
                continue
            record["text"] = record["text"].replace("\\n", " ")
            record["edition"] = None if record["edition"] == "-" else record["edition"]
            record["date"] = parse_datetime(record["date"])
            if "host" not in record:
                record["host"] = urlsplit(record["url"]).netloc

            yield record


def read_parsers_csv_dir(directory):
    for record in parse_dir(directory, ".csv", read_parsers_csv, print_interval=1000):
        yield record
