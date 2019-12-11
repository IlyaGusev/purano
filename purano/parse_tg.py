import argparse
import os
from collections import namedtuple
import xml.etree.cElementTree as ET
import pandas as pd

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
    return doc


def parse_dir(directory):
    for r, d, f in os.walk(directory):
        for file_name in f:
            file_name = os.path.join(r, file_name)
            if not file_name.endswith(".html"):
                continue
            try:
                yield parse_xml(file_name)
            except Exception as e:
                print(e)
                continue


def parse_tg(directory, ndocs, save_fields):
    documents = []
    save_fields = save_fields.split(",")
    for i, document in enumerate(parse_dir(directory)):
        if i >= ndocs:
            break
        documents.append(document)
    df = pd.DataFrame(documents)
    df = df[save_fields]
    print(df.head(5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--ndocs", type=int, default=None)
    parser.add_argument("--save-fields", type=str, default="url,title,text,authors")
    args = parser.parse_args()
    parse_tg(**vars(args))
