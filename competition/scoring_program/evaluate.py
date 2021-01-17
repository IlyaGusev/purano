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
        for line in truth:
            first_url, second_url, answer = line.strip().split("\t")
            answers[(first_url, second_url)] = answer

    predictions = dict()
    with open(submission_filename, "r") as submission:
        for line in submission:
            first_url, second_url, answer = line.strip().split("\t")
            predictions[(first_url, second_url)] = answer

    p_size = len(predictions)
    a_size = len(answers)
    assert p_size == a_size, "Wrong number of predictions: {} vs {}".format(p_size, a_size)

    correct_count = 0
    for sample_id, answer in answers.items():
        assert sample_id in predictions, "{} is missing".format(sample_id)
        prediction = predictions[sample_id]
        if prediction == answer:
            correct_count += 1
    accuracy = float(correct_count) / a_size

    with open(output_filename, 'wb') as output:
        output.write("accuracy:{}".format(accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()
    evaluate(**vars(args))

