import json
import os
import argparse

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from purano.models import Document


def entity_to_tuple(e, text):
    result = (e.begin, e.end, e.tag, text[e.begin: e.end])
    return result


def to_jsonl(input_file, output_file):
    assert os.path.exists(input_file)
    db_engine = "sqlite:///{}".format(input_file)
    engine = create_engine(db_engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    query = session.query(Document)
    query = query.join(Document.info)
    docs = query.all()
    with open(output_file, "w") as w:
        for doc in docs:
            title = doc.title.replace("\xa0", " ").strip()
            text = doc.text.replace("\xa0", " ").strip()
            w.write(json.dumps({
                "title": title,
                "text": text,
                "url": doc.url,
                "timestamp": int(doc.date.timestamp()),
                "host": doc.host,
                "embedding": list(doc.info["gen_title_embedding"]),
                "keywords": list(doc.info["tfidf_keywords"]),
                "title_entities": [entity_to_tuple(e, doc.title) for e in doc.info["title_slovnet_ner"]],
                "text_entities": [entity_to_tuple(e, doc.text) for e in doc.info["text_slovnet_ner"]]
            }, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()
    to_jsonl(**vars(args))
