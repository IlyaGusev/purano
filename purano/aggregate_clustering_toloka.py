import argparse
from collections import Counter
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score

from purano.io import write_markup_tsv


def main(answers_file_name, min_proba, target_file_name):
    with open(answers_file_name, "r") as r:
        header = tuple(next(r).strip().split("\t"))
        records = []
        for line in r:
            fields = line.strip().split("\t")
            assert len(fields) == len(header), fields
            records.append(dict(zip(header, fields)))
    # Filter honeypots out
    records = [r for r in records if not r["GOLDEN:quality"]]

    # Normalize fields
    for r in records:
        r.pop("GOLDEN:quality", None)
        r.pop("HINT:text", None)
        for key, value in list(r.items()):
            new_key = key.split(":")[-1]
            r[new_key] = r.pop(key)
        r.pop("link", None)
        r.pop("assignment_id", None)
        r.pop("status", None)
        r.pop("started", None)

    # Calc inter-annotator agreement
    annotator2labels = defaultdict(dict)
    unique_keys = list(set([(r["first_url"], r["second_url"]) for r in records]))
    unique_workers = list(set([r["worker_id"] for r in records]))
    unique_res = list(set([r["quality"] for r in records]))
    res2num = {res: i for i, res in enumerate(unique_res)}
    for r in records:
        annotator2labels[r["worker_id"]][(r["first_url"], r["second_url"])] = r["quality"]
    worker2labels = {}
    for worker_id in unique_workers:
        worker_labels = []
        worker_res = annotator2labels[worker_id]
        for key in unique_keys:
            if key not in worker_res:
                worker_labels.append(-1)
                continue
            worker_labels.append(res2num[worker_res[key]])
        worker2labels[worker_id] = worker_labels
    scores = []
    for w1, labels1 in worker2labels.items():
        for w2, labels2 in worker2labels.items():
            if w1 == w2:
                continue
            fixed_labels1 = []
            fixed_labels2 = []
            for l1, l2 in zip(labels1, labels2):
                if l1 == -1 or l2 == -1:
                    continue
                fixed_labels1.append(l1)
                fixed_labels2.append(l2)
            if fixed_labels1 and fixed_labels2:
                score = cohen_kappa_score(fixed_labels1, fixed_labels2)
                if -1.0 <= score <= 1.0:
                    scores.append(score)
    print("Avg kappa score: {}".format(sum(scores)/len(scores)))

    results = defaultdict(list)
    for r in records:
        results[(r["first_url"], r["second_url"])].append(r["quality"])

    data = {(r["first_url"], r["second_url"]): r for r in records}
    proba_count = Counter()
    print("Bad examples: ")
    for key, res in results.items():
        res_count = Counter(res)
        votes_count = len(res)
        votes_for_win = res_count.most_common(1)[0][1]
        proba = votes_for_win / votes_count
        proba_count[proba] += 1
        if proba < min_proba:
            print("Key:", key)
            print("Answers:", ", ".join(res))
            data.pop(key)
            continue
        data[key]["quality"] = res_count.most_common(1)[0][0]
        data[key].pop("worker_id")
    print("Votes for majority: ")
    for proba, sample_count in proba_count.items():
        print("{}: {}".format(proba, sample_count))
    write_markup_tsv([r for _, r in data.items()], target_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file-name", type=str, required=True)
    parser.add_argument("--min-proba", type=float, default=0.7)
    parser.add_argument("--target-file-name", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
