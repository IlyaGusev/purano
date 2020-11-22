import os
import shutil
import string

from pyonmttok import Tokenizer
from pymorphy2 import MorphAnalyzer

tokenizer = Tokenizer("conservative", joiner_annotate=False)
morph = MorphAnalyzer()

def parse_dir(directory, ext, parse_file_func, print_interval=None):
    documents_count = 0
    for r, d, f in os.walk(directory):
        for file_name in f:
            file_name = os.path.join(r, file_name)
            if not file_name.endswith(ext):
                continue
            try:
                for record in parse_file_func(file_name):
                    yield record
                    documents_count += 1
                    if print_interval and documents_count % print_interval == 0:
                        print("Parsed {} documents".format(documents_count))
            except Exception as e:
                print(e)
                continue


def tokenize(text, lower=True):
    text = str(text).strip().replace("\n", " ").replace("\xa0", " ")
    if lower:
        text = text.lower()
    tokens, _ = tokenizer.tokenize(text)
    return tokens


def tokenize_to_lemmas(text):
    tokens = tokenize(text)
    tokens = filter(lambda x: x not in string.punctuation, tokens)
    tokens = filter(lambda x: not x.isnumeric(), tokens)
    tokens = filter(lambda x: len(x) >= 2, tokens)
    tokens = [morph.parse(token)[0].normal_form for token in tokens]
    return tokens


def get_true_file(file_path):
    if file_path.endswith(".tar.gz"):
        true_file_path = file_path.replace(".tar.gz", "")
        if not os.path.exists(true_file_path):
            dir_path = os.path.dirname(os.path.realpath(file_path))
            shutil.unpack_archive(file_path, dir_path)
        file_path = true_file_path
    return file_path

