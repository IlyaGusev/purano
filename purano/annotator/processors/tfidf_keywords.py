import os
from typing import List
from collections import Counter

import numpy as np

from purano.annotator.processors import Processor
from purano.models import Document
from purano.proto.info_pb2 import Info as InfoPb
from purano.util import tokenize_to_lemmas

@Processor.register("tfidf_keywords")
class TfIdfKeywordsProcessor(Processor):
    def __init__(self,
        idfs_vocabulary: str,
        top_k: int
    ):
        self.top_k = top_k
        self.idfs = dict()
        with open(idfs_vocabulary, "r") as r:
            for line in r:
                word, idf = line.strip().split("\t")
                self.idfs[word] = float(idf)

    def __call__(self,
        docs: List[Document],
        infos: List[InfoPb],
        input_fields: List[str],
        output_field: str,
    ):
        for doc, info in zip(docs, infos):
            tfidfs = []
            tokens = [token for field in input_fields for token in tokenize_to_lemmas(getattr(doc, field))]
            freqs = Counter(tokens)
            tfs = {token: float(f) / len(tokens) for token, f in freqs.items()}
            for token, tf in tfs.items():
                idf = self.idfs.get(token, 0.0)
                tfidf = tf * idf
                tfidfs.append((tf * idf, token))
            tfidfs.sort()
            keywords = [token for tfidf, token in tfidfs[-self.top_k:]]
            getattr(info, output_field).extend(keywords)
