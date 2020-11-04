import os
from typing import List

import numpy as np
import fasttext
import pyonmttok

from purano.annotator.processors import Processor
from purano.models import Document
from purano.proto.info_pb2 import Info as InfoPb

@Processor.register("fasttext")
class FasttextProcessor(Processor):
    def __init__(self, path: str):
        self.model_path = path
        self.model = fasttext.load_model(path)
        self.tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=False)

    def __call__(self,
        docs: List[Document],
        infos: List[InfoPb],
        input_fields: List[str],
        output_field: str,
        max_tokens_count: int,
        agg_type: str
    ):
        assert agg_type in ("mean", "max", "mean||max")
        word_embeddings = self.calc_word_embeddings(docs, input_fields, max_tokens_count)
        mean_embeddings = np.mean(word_embeddings, axis=1)
        max_embeddings = np.max(word_embeddings, axis=1)
        if agg_type == "mean":
            return mean_embeddings
        if agg_type == "max":
            return max_embeddings
        assert agg_type == "mean||max"
        sample_embeddings = np.concatenate((mean_embeddings, max_embeddings), axis=1)
        for doc_num, info in enumerate(infos):
            getattr(info, output_field).extend(sample_embeddings[doc_num])

    def preprocess(self, text):
        text = str(text).strip().replace("\n", " ").replace("\xa0", " ").lower()
        tokens, _ = self.tokenizer.tokenize(text)
        return tokens

    def calc_word_embeddings(self, docs, input_fields, max_tokens_count):
        batch_size = len(docs)
        word_embeddings = np.zeros((batch_size, max_tokens_count, self.model.get_dimension()), dtype=np.float64)
        real_max_tokens_count = 0
        for doc_num, doc in enumerate(docs):
            sample = " ".join([getattr(doc, input_field) for input_field in input_fields])
            tokens = self.preprocess(sample)[:max_tokens_count]
            real_max_tokens_count = max(real_max_tokens_count, len(tokens))
            for token_num, token in enumerate(tokens):
                word_embeddings[doc_num, token_num, :] = self.model.get_word_vector(token)
        word_embeddings = word_embeddings[:, :real_max_tokens_count, :]
        return word_embeddings

