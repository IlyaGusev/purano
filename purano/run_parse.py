import argparse
import os
import json
from typing import Optional

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from fasttext import load_model as ft_load_model
from pandas import DataFrame
from sqlalchemy import create_engine
from pyonmttok import Tokenizer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from purano.models import Document, Base
from purano.io import read_tg_html_dir, read_tg_jsonl_dir, \
    read_tg_jsonl, read_parsers_csv_dir, read_parsers_csv

FASTTEXT_LABEL_OFFSET = len("__label__")


class DocumentsCleaner:
    def __init__(self, config_path):
        self.config = json.loads(jsonnet_evaluate_file(config_path))
        self.lang_detect_model_path = self.config["lang_detect_model_path"]
        self.cat_detect_model_path = self.config["cat_detect_model_path"]
        self.max_tokens = self.config.get("max_tokens")
        self.is_lower = self.config["is_lower"]
        self.is_russian_only = self.config.get("is_russian_only", False)
        self.is_news_only = self.config.get("is_news_only", False)
        assert os.path.exists(self.lang_detect_model_path), "No language detection model found"
        assert os.path.exists(self.cat_detect_model_path), "No category detection model found"
        self.lang_detect_model = ft_load_model(self.lang_detect_model_path)
        self.cat_detect_model = ft_load_model(self.cat_detect_model_path)
        self.tokenizer = Tokenizer("conservative", joiner_annotate=False)

    def preprocess(self, text):
        text = str(text).strip().replace("\n", " ").replace("\xa0", " ")
        if self.is_lower:
            text = text.lower()
        tokens, _ = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        return " ".join(tokens)

    def __call__(self, document) -> Optional[Document]:
        title = document.get("title")
        description = document.get("description", "")
        text = document.get("text")

        lang_text = text[:100]
        lang_text_sample = " ".join((title, description, lang_text))
        lang_text_sample = lang_text_sample.replace("  ", " ").replace("\n", " ")
        (lang_label,), (lang_prob,) = self.lang_detect_model.predict(lang_text_sample, k=1)
        lang_label = lang_label[FASTTEXT_LABEL_OFFSET:]
        document["language"] = lang_label
        if self.is_russian_only and lang_label != "ru" or lang_prob < 0.6:
            return None

        document["patched_title"] = self.preprocess(title)
        document["patched_text"] = self.preprocess(text)

        cat_text_sample = document["patched_title"] + " " + document["patched_text"]
        cat_text_sample = self.preprocess(cat_text_sample)
        (cat_label,), (cat_prob,) = self.cat_detect_model.predict(cat_text_sample, k=1)
        cat_label = cat_label[FASTTEXT_LABEL_OFFSET:]
        document["category"] = cat_label
        if self.is_news_only and cat_label == "not_news":
            return None
        return document


def run_parse(
    inputs,
    fmt,
    output_file,
    ndocs,
    save_fields,
    start_date,
    end_date,
    cleaner_config,
    log_rate,
    output_jsonl_path
):
    # Choose right parser
    parser = None
    if fmt == "html":
        parser = read_tg_html_dir
    elif fmt == "jsonl" and os.path.isdir(inputs):
        parser = read_tg_jsonl_dir
    elif fmt == "jsonl" and os.path.isfile(inputs):
        parser = read_tg_jsonl
    elif fmt == "csv" and os.path.isdir(inputs):
        parser = read_parsers_csv_dir
    elif fmt == "csv" and os.path.isfile(inputs):
        parser = read_parsers_csv
    else:
        assert False, "Parser for format {} is not set".format(fmt)

    existing_urls = set()
    if os.path.exists(output_file):
        db_engine = "sqlite:///{}".format(output_file)
        engine = create_engine(db_engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        query = session.query(Document)
        docs = query.all()
        for doc in docs:
            existing_urls.add(doc.url)

    # Parse and clean documents
    cleaner = DocumentsCleaner(cleaner_config)
    documents = {}
    for i, document in enumerate(parser(inputs)):
        if ndocs and i >= ndocs:
            break
        if document["url"] in existing_urls:
            continue
        document = cleaner(document)
        if document:
            documents[document["url"]] = document
            if log_rate and len(documents) % log_rate == 0:
                print("{} documents processed".format(len(documents)))
    documents = documents.values()

    # Filter, undup, sort
    df = DataFrame(documents)
    df = df[save_fields]
    df = df[(~df["text"].isnull() & ~df["title"].isnull())]
    df.drop_duplicates(subset=["url", "title", "text"], keep="last", inplace=True)
    df.drop_duplicates(subset=["url"], keep="last", inplace=True)
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] < end_date]
    df.sort_values("date", inplace=True)
    print("{} will be saved".format(len(df)))

    # Print dataset info
    print(df.info())
    print(df.head(5))

    if output_jsonl_path:
        df.to_json(output_jsonl_path, orient="records", lines=True, force_ascii=False)

    # Save to database
    db_engine = "sqlite:///{}".format(output_file)
    engine = create_engine(db_engine)
    Base.metadata.create_all(engine, Base.metadata.tables.values(), checkfirst=True)
    df.to_sql(Document.__tablename__, engine.raw_connection(), if_exists="append", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, required=True)
    parser.add_argument("--ndocs", type=int, default=None)
    parser.add_argument("--output-file", type=str, default="output/parsed.db")
    parser.add_argument("--save-fields", type=str,
                        default="url,host,title,text,date,patched_title,patched_text,category")
    parser.add_argument("--cleaner-config", type=str, default="configs/cleaner.jsonnet")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--fmt", type=str, choices=("html", "jsonl", "csv"), required=True)
    parser.add_argument("--log-rate", type=int, default=None)
    parser.add_argument("--output-jsonl-path", type=str, default=None)
    args = parser.parse_args()
    args.save_fields = args.save_fields.split(",")
    run_parse(**vars(args))
