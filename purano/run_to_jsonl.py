import json
import os
import argparse

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from purano.models import Document


def to_jsonl(input_file, output_file):
    assert os.path.exists(input_file)
    db_engine = "sqlite:///{}".format(input_file)
    engine = create_engine(db_engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    query = session.query(Document)
    docs = query.all()
    with open(output_file, "w") as w:
        for doc in docs:
            title = doc.title.replace("\xa0", " ").strip()
            text = doc.text.replace("\xa0", " ").strip()
            w.write(json.dumps({
                "title": title,
                "text": text,
                "url": doc.url,
                "timestamp": int(doc.date.timestamp())
            }, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()
    to_jsonl(**vars(args))
