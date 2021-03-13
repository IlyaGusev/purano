#!/usr/bin/env python3
import razdel
from rouge_metric import PyRouge
from nltk.translate.bleu_score import corpus_bleu

import argparse
import os
import json
from collections import defaultdict


def calc_metrics(refs, hyps, metric="all"):
    metrics = dict()
    metrics["count"] = len(hyps)
    metrics["ref_example"] = refs[-1][-1]
    metrics["hyp_example"] = hyps[-1]
    if metric in ("bleu", "all"):
        metrics["bleu"] = corpus_bleu(refs, hyps)
    if metric in ("rouge", "all"):
        rouge = PyRouge(rouge_l=True, multi_ref_mode="best")
        scores = rouge.evaluate(hyps, refs)
        metrics.update(scores)
    return metrics


def postprocess(refs, hyp, tokenize_after, lower):
    refs = [ref.strip() for ref in refs]
    hyp = hyp.strip()
    if tokenize_after:
        hyp = " ".join([token.text for token in razdel.tokenize(hyp)])
        refs = [" ".join([token.text for token in razdel.tokenize(ref)]) for ref in refs]
    if lower:
        hyp = hyp.lower()
        refs = [ref.lower() for ref in refs]
    return refs, hyp


def get_final_metrics(hyps, all_refs, tokenize_after=True, lower=True):
    clean_hyps = []
    clean_refs = []
    for hyp, refs in zip(hyps, all_refs):
        refs, hyp = postprocess(refs, hyp, tokenize_after, lower)
        assert hyp
        clean_refs.append(refs)
        clean_hyps.append(hyp)
    metrics = calc_metrics(clean_refs, clean_hyps)
    r1 = float(metrics["rouge-1"]["f"])
    r2 = float(metrics["rouge-2"]["f"])
    rl = float(metrics["rouge-l"]["f"])
    bleu = float(metrics["bleu"])
    return {
        "rouge": (r1 + r2 + rl) / 3,
        "bleu": bleu,
        "rouge-1": r1,
        "rouge-2": r2,
        "rouge-l": rl
    }


def evaluate(input_dir, output_dir):
    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)
        return

    if not os.path.isdir(truth_dir):
        print("%s doesn't exist" % truth_dir)
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, "scores.txt")
    truth_filename = os.path.join(truth_dir, "truth.txt")
    submission_filename = os.path.join(submit_dir, "answer.txt")

    id2refs = defaultdict(list)
    with open(truth_filename, "r") as truth:
        for line in truth:
            record = json.loads(line)
            sample_id = record["sample_id"]
            id2refs[sample_id].append(record["title"])

    id2hyps = dict()
    with open(submission_filename, "r") as submission:
        for line in submission:
            record = json.loads(line)
            sample_id = record["sample_id"]
            id2hyps[sample_id] = record["title"]

    p_size = len(id2hyps)
    a_size = len(id2refs)
    assert p_size == a_size, "Wrong number of predictions: {} vs {}".format(p_size, a_size)

    all_hyps = []
    all_refs = []
    for sample_id, hyp in id2hyps.items():
        all_refs.append(id2refs[sample_id])
        all_hyps.append(hyp)

    metrics = get_final_metrics(all_hyps, all_refs)
    with open(output_filename, 'w') as output:
        output.write("rouge:{}\n".format(metrics["rouge"]))
        output.write("bleu:{}\n".format(metrics["bleu"]))
        output.write("rouge_1:{}\n".format(metrics["rouge-1"]))
        output.write("rouge_2:{}\n".format(metrics["rouge-2"]))
        output.write("rouge_l:{}".format(metrics["rouge-l"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()
    evaluate(**vars(args))
