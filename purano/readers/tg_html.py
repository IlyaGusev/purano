import os

from dateutil.parser import parse as parse_datetime
from urllib.parse import urlsplit
from xml.etree.cElementTree import parse as parse_xml

from purano.util import parse_dir

def parse_tg_html(file_name):
    tree = parse_xml(file_name)
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

    def parse_links(element):
        links = []
        for elem in element:
            if elem.tag == "a" and elem.get("href"):
                links.append(elem.get("href"))
                continue
            links += parse_links(elem)
        return links

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
    doc["date"] = parse_datetime(doc["published_time"])
    doc["host"] = urlsplit(doc["url"]).netloc

    yield doc


def parse_tg_html_dir(directory):
    for record in parse_dir(directory, ".html", parse_tg_html, print_interval=1000):
        yield record
