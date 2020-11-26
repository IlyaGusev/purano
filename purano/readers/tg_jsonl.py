import os
import json
from datetime import datetime
from urllib.parse import urlsplit

from purano.util import parse_dir


def parse_tg_jsonl(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            assert "title" in record
            assert "text" in record
            assert "url" in record
            timestamp = record.get("timestamp")
            record["date"] = datetime.utcfromtimestamp(timestamp)
            if "host" not in record:
                record["host"] = urlsplit(record["url"]).netloc
            yield record


def parse_tg_jsonl_dir(directory):
    for record in parse_dir(directory, ".jsonl", parse_tg_jsonl):
        yield record
