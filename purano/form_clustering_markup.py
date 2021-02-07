import argparse
import json
import random

from purano.io import read_markup_tsv, write_markup_tsv


def main(
    original_jsonl,
    threads_json,
    honey_tsv,
    current_markup_tsv,
    output_tsv,
    true_prob,
    random_prob
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

    honey_records = read_markup_tsv(honey_tsv)

    with open(threads_json, "r") as r:
        threads = json.load(r)
        random.shuffle(threads)

    markup_keys = []

    def add_key(key):
        if key in existing_urls:
            return
        markup_keys.append(key)
        existing_urls.add(key)
        existing_urls.add((key[1], key[0]))

    prev_thread_url = None
    for thread in threads:
        thread_urls = list(set(thread["articles"]))
        random.shuffle(thread_urls)
        first_url = thread_urls[0]
        for second_url in thread_urls[1:]:
            key = (first_url, second_url)
            if random.random() < true_prob:
                add_key(key)
        key = (prev_thread_url, first_url)
        if prev_thread_url and random.random() < random_prob:
            add_key(key)
        prev_thread_url = first_url

    markup = []
    for url1, url2 in markup_keys:
        first = url2record[url1]
        second = url2record[url2]
        markup.append({
            "first_url": url1,
            "second_url": url2,
            "first_title": first["title"],
            "second_title": second["title"],
            "first_text": first["text"],
            "second_text": second["text"]
        })

    markup_len = len(honey_records) * 9
    random.shuffle(markup)
    final_markup = markup[:markup_len] + honey_records
    random.shuffle(final_markup)
    print(len(final_markup))

    write_markup_tsv(final_markup, output_tsv, input_prefix="INPUT:", res_prefix="GOLDEN:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-jsonl", type=str, required=True)
    parser.add_argument("--threads-json", type=str, required=True)
    parser.add_argument("--honey-tsv", type=str, required=True)
    parser.add_argument("--current-markup-tsv", type=str, default=None)
    parser.add_argument("--output-tsv", type=str, required=True)
    parser.add_argument("--true-prob", type=float, default=0.5)
    parser.add_argument("--random-prob", type=float, required=0.01)

    args = parser.parse_args()
    main(**vars(args))
