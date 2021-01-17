import json
from datetime import datetime
from urllib.parse import urlsplit

from purano.io.parse_dir import parse_dir


def read_tg_jsonl(file_name):
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
            record["text"] = record.pop("text").strip().replace("\xa0", " ")
            record["title"] = record.pop("title").strip().replace("\xa0", " ")
            yield record


def read_tg_jsonl_dir(directory):
    for record in parse_dir(directory, ".jsonl", read_tg_jsonl):
        yield record
