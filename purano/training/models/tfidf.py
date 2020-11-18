import string
import json
from collections import Counter

import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer

from purano.readers.tg_jsonl import parse_tg_jsonl
from purano.util import tokenize_to_lemmas

corpus = []
for record in parse_tg_jsonl("data/documents/ru_tg_0511_0517.jsonl"):
    corpus.append(record["text"])
    corpus.append(record["title"])

vectorizer = TfidfVectorizer(tokenizer=tokenize_to_lemmas, max_df=0.2, min_df=20)
vectorizer.fit(corpus)
idf_vector = vectorizer.idf_.tolist()
idfs = list()
for word, idx in vectorizer.vocabulary_.items():
    idfs.append((idf_vector[idx], word))

idfs.sort()
with open("idfs.txt", "w") as w:
    for idf, word in idfs:
        w.write("{}\t{}\n".format(word, idf))
