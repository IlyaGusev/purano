#!/usr/bin/env python

import argparse
import sys
import os


def evaluate(input_dir, output_dir):
    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print "%s doesn't exist" % submit_dir
        return

    if not os.path.isdir(truth_dir):
        print "%s doesn't exist" % truth_dir
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, "scores.txt")
    truth_filename = os.path.join(truth_dir, "truth.txt")
    submission_filename = os.path.join(submit_dir, "answer.txt")

    answers = dict()
    with open(truth_filename, "r") as truth:
        header = next(truth).strip().split("\t")
        for line in truth:
            fields = line.strip().split("\t")
            record = dict(zip(header, fields))
            if record["dataset"] == "0527":
                answers[(record["first_url"], record["second_url"])] = record["quality"]

    predictions = dict()
    with open(submission_filename, "r") as submission:
        header = next(submission).strip().split("\t")
        for line in submission:
            fields = line.strip().split("\t")
            record = dict(zip(header, fields))
            if record["dataset"] == "0527":
                predictions[(record["first_url"], record["second_url"])] = record["quality"]

    p_size = len(predictions)
    a_size = len(answers)
    assert p_size == a_size, "Wrong number of predictions: {} vs {}".format(p_size, a_size)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for sample_id, answer in answers.items():
        assert sample_id in predictions, "{} is missing".format(sample_id)
        prediction = predictions[sample_id]
        if answer == "OK":
            if prediction == answer:
                tp += 1
            else:
                fn += 1
        if answer == "BAD":
            if prediction == answer:
                tn += 1
            else:
                fp += 1
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    with open(output_filename, 'wb') as output:
        output.write("f1_score:{}".format(f1_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()
    evaluate(**vars(args))

