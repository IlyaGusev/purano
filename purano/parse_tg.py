import argparse
import os
from collections import namedtuple
from urllib.parse import urlsplit
from datetime import datetime
import xml.etree.cElementTree as ET

import fasttext
import dateutil.parser
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from purano.models import Document, Base, Agency


def parse_links(element):
    links = []
    for elem in element:
        if elem.tag == "a" and elem.get("href"):
            links.append(elem.get("href"))
            continue
        links += parse_links(elem)
    return links


def parse_xml(file_name):
    tree = ET.parse(file_name)
    root = tree.getroot()
    head_element = root.find("head")
    doc = dict()
    for meta_element in head_element.iterfind("meta"):
        prop = meta_element.get("property")
        content = meta_element.get("content")
        if not prop or not content:
            continue
        if prop == "og:title":
            doc["title"] = content
        elif prop == "og:url":
            doc["url"] = content
        elif prop == "og:site_name":
            doc["site_name"] = content
        elif prop == "og:description":
            doc["description"] = content
        elif prop == "article:published_time":
            doc["published_time"] = content
    body_element = root.find("body")
    article_element = body_element.find("article")
    text = ""
    links = []
    for p_element in article_element.iterfind("p"):
        text += " ".join(p_element.itertext()) + " "
        links += parse_links(p_element)
    doc["text"] = text
    address_element = article_element.find("address")
    doc["text_time"] = None
    doc["authors"] = None
    if address_element:
        time_element = address_element.find("time")
        if time_element is not None and time_element.get("datetime"):
            doc["text_time"] = time_element.get("datetime")
        author_element = address_element.find("a")
        if author_element is not None and author_element.get("rel") == "author":
            doc["authors"] = author_element.text
    doc["date"] = dateutil.parser.parse(doc["published_time"])
    return doc


def parse_dir(directory):
    documents_count = 0
    for r, d, f in os.walk(directory):
        for file_name in f:
            file_name = os.path.join(r, file_name)
            if not file_name.endswith(".html"):
                continue
            try:
                yield parse_xml(file_name)
                documents_count += 1
                if documents_count % 1000 == 0:
                    print("Parsed {} documents".format(documents_count))
            except Exception as e:
                print(e)
                continue


def parse_tg(db_engine, directory, ndocs, save_fields, lang_detect_model, news_detect_model):
    engine = create_engine(db_engine)
    Base.metadata.create_all(engine, Base.metadata.tables.values(),checkfirst=True)

    documents = {}
    save_fields = save_fields.split(",")
    lang_detect_model = fasttext.load_model(lang_detect_model)
    news_detect_model = fasttext.load_model(news_detect_model)
    for i, document in enumerate(parse_dir(directory)):
        if ndocs and i >= ndocs:
            break
        text_sample = document["title"] + " " + document["text"][:200]
        text_sample = text_sample.replace("\n", " ")
        language_predictions = lang_detect_model.predict(text_sample, k=1)
        language = language_predictions[0][0][9:]
        probability = language_predictions[1][0]
        if language != "ru" or probability < 0.7:
            continue
        news_predictions = news_detect_model.predict(text_sample, k=1)
        is_news = news_predictions[0][0][9:] == "news"
        probability = news_predictions[1][0]
        if not is_news and probability > 0.5:
            continue
        documents[document["url"]] = document
    documents = documents.values()
    print("{} will be saved".format(len(documents)))

    hosts = set()
    for document in documents:
        host = urlsplit(document["url"]).netloc
        document["host"] = host
        hosts.add(host)

    Session = sessionmaker(bind=engine)
    session = Session()
    existing_agencies = {agency.host for agency in session.query(Agency).all()}
    for host in hosts:
        if host in existing_agencies:
            continue
        agency = Agency()
        agency.host = host
        session.add(agency)
    session.commit()

    host_to_agency_id = {agency.host: agency.id for agency in session.query(Agency).all()}
    for document in documents:
        document["agency_id"] = host_to_agency_id[document["host"]]
        document["edition"] = document.pop("host")

    df = pd.DataFrame(documents)
    df = df[save_fields]
    print(df.head(5))
    df.to_sql(Document.__tablename__, engine.raw_connection(), if_exists='append', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-engine", type=str, default="sqlite:///news.db")
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--ndocs", type=int, default=None)
    parser.add_argument("--save-fields", type=str, default="url,title,text,authors,date,edition,agency_id")
    parser.add_argument("--lang-detect-model", type=str, default="models/lang_detect.ftz")
    parser.add_argument("--news-detect-model", type=str, default="models/ru_news_detect.ftz")
    args = parser.parse_args()
    parse_tg(**vars(args))
