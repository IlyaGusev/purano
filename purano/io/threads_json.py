import json


def read_threads_json(file_name):
    threads = []
    with open(file_name, "r") as r:
        threads = json.load(r)
    labels = dict()
    for label, thread in enumerate(threads):
        thread_urls = {url for url in thread["articles"]}
        for url in thread_urls:
            labels[url] = label
    return labels
