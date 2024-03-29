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
            if "timestamp" in record:
                timestamp = record.get("timestamp")
                record["date"] = datetime.utcfromtimestamp(timestamp)
            elif "date" in record:
                record["date"] = datetime.strptime(record["date"], "%Y-%m-%d")
            if "host" not in record:
                record["host"] = urlsplit(record["url"]).netloc
            record["text"] = record.pop("text").strip().replace("\xa0", " ")
            record["title"] = record.pop("title").strip().replace("\xa0", " ")
            if "description" in record and record["description"]:
                record["description"] = record.pop("description").strip().replace("\xa0", " ")
            if "out_links" not in record:
                record["out_links"] = "[]"
            yield record


def read_tg_jsonl_dir(directory):
    for record in parse_dir(directory, ".jsonl", read_tg_jsonl):
        yield record
