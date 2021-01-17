import argparse
import copy
import json
import itertools
import random

from purano.util.markup import read_markup_tsv, write_markup_tsv


def main(
    original_jsonl,
    threads_json,
    honey_tsv,
    current_markup_tsv,
    output_tsv,
    true_prob=0.5,
    random_prob=0.01
):
    url2record = dict()
    with open(original_jsonl, "r") as r:
        for line in r:
            record = json.loads(line)
            url2record[record["url"]] = record

    existing_urls = set()
    if current_markup_tsv:
        current_markup = read_markup_tsv(current_markup_tsv)
        existing_urls = {(r["first_url"], r["second_url"]) for r in current_markup}
        existing_urls |= {(r["second_url"], r["first_url"]) for r in current_markup}

    # honey_records = read_markup_tsv(honey_tsv)

    with open(threads_json, "r") as r:
        threads = json.load(r)

    markup = []

    def add_key(key, context):
        if key in existing_urls:
            return
        markup.append((key, context))
        existing_urls.add(key)
        existing_urls.add((key[1], key[0]))

    for thread in threads:
        thread_urls = list(set(thread["articles"]))
        thread_urls = thread_urls[:7]
        if len(thread_urls) <= 4:
            continue
        for url1, url2 in itertools.combinations(thread_urls, 2):
            add_key((url1, url2), copy.copy(thread_urls))
            print(url1, url2)

    final_markup = []
    for (url1, url2), context in markup:
        first = url2record[url1]
        second = url2record[url2]
        if random.random() < 0.5:
            first, second = second, first
        context.remove(url1)
        context.remove(url2)
        markup_record = {
            "first_url": first["url"],
            "second_url": second["url"],
            "first_title": first["title"],
            "second_title": second["title"],
            "first_text": first["text"],
            "second_text": second["text"]
        }
        for i, url in enumerate(context):
            context_title = url2record[url]["title"]
            markup_record["context_title_{}".format(i)] = context_title
        final_markup.append(markup_record)

    # markup_len = len(honey_records) * 9
    markup_len = 1000
    random.shuffle(final_markup)
    final_markup = final_markup[:markup_len]
    # final_markup = markup[:markup_len] + honey_records
    # random.shuffle(final_markup)
    print(len(final_markup))

    write_markup_tsv(final_markup, output_tsv, res_key="res")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-jsonl", type=str, required=True)
    parser.add_argument("--threads-json", type=str, required=True)
    parser.add_argument("--honey-tsv", type=str, default=None)
    parser.add_argument("--current-markup-tsv", type=str, default=None)
    parser.add_argument("--output-tsv", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
