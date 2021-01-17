import os
import shutil

from purano.util.tokenization import tokenize, tokenize_to_lemmas


def get_true_file(file_path):
    if file_path.endswith(".tar.gz"):
        true_file_path = file_path.replace(".tar.gz", "")
        if not os.path.exists(true_file_path):
            dir_path = os.path.dirname(os.path.realpath(file_path))
            shutil.unpack_archive(file_path, dir_path)
        file_path = true_file_path
    return file_path
