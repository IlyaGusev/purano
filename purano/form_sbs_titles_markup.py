import argparse
import copy
import json
import itertools
import random

from purano.io import read_markup_tsv, write_markup_tsv


def main(
    original_jsonl,
    threads_json,
    honey_tsv,
    current_markup_tsv,
    output_tsv,
    clustering_markup_tsv,
    include_bad_samples,
    randomize=False
):
    url2record = dict()
    with open(original_jsonl, "r") as r:
        for line in r:
            record = json.loads(line)
            url2record[record["url"]] = record

    existing_keys = set()
    if current_markup_tsv:
        current_markup = read_markup_tsv(current_markup_tsv)
        existing_keys = {(r["left_url"], r["right_url"]) for r in current_markup}
        existing_keys |= {(r["right_url"], r["left_url"]) for r in current_markup}

    existing_urls = set()
    if clustering_markup_tsv:
        clustering_markup = read_markup_tsv(clustering_markup_tsv)
        for r in clustering_markup:
            existing_urls.add(r["first_url"])
            existing_urls.add(r["second_url"])

    honey_records = read_markup_tsv(honey_tsv)

    with open(threads_json, "r") as r:
        threads = json.load(r)

    markup = []

    def add_key(key, context):
        if key in existing_keys:
            return
        markup.append((key, context))
        existing_keys.add(key)
        existing_keys.add((key[1], key[0]))

    prev_thread_urls = None
    for thread in threads:
        thread_urls = set(thread["articles"])
        thread_urls = [url for url in thread_urls if url not in existing_urls]
        thread_urls = thread_urls[:7]
        if len(thread_urls) <= 4:
            continue
        for url1, url2 in itertools.combinations(thread_urls, 2):
            add_key((url1, url2), copy.copy(thread_urls))
        if include_bad_samples != 0 and prev_thread_urls:
            for _ in range(include_bad_samples):
                prev_url = random.choice(prev_thread_urls)
                cur_url = random.choice(thread_urls)
                add_key((prev_url, cur_url), copy.copy(thread_urls))
        prev_thread_urls = thread_urls

    final_markup = []
    bad_count = 0
    for (url1, url2), context in markup:
        first = url2record[url1]
        second = url2record[url2]
        if random.random() < 0.5:
            first, second = second, first
        if url1 in context:
            context.remove(url1)
        else:
            bad_count += 1
        if url2 in context:
            context.remove(url2)
        else:
            bad_count += 1
        markup_record = {
            "left_url": first["url"],
            "right_url": second["url"],
            "left_title": first["title"],
            "right_title": second["title"]
        }
        markup_record["info"] = json.dumps([url2record[url]["title"] for url in context], ensure_ascii=False)
        final_markup.append(markup_record)

    print("Bad count: ", bad_count)
    print("All count: ", len(final_markup))
    markup_len = len(honey_records) * 9
    if randomize:
        random.shuffle(final_markup)
    final_markup = final_markup[:markup_len] + honey_records
    random.shuffle(final_markup)
    print(len(final_markup))

    write_markup_tsv(final_markup, output_tsv, res_key="result", res_prefix="GOLDEN:", input_prefix="INPUT:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-jsonl", type=str, required=True)
    parser.add_argument("--threads-json", type=str, required=True)
    parser.add_argument("--honey-tsv", type=str, default=None)
    parser.add_argument("--current-markup-tsv", type=str, default=None)
    parser.add_argument("--clustering-markup-tsv", type=str, default=None)
    parser.add_argument("--output-tsv", type=str, required=True)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--include-bad-samples", type=int, default=0)
    args = parser.parse_args()
    main(**vars(args))
