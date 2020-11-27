from typing import List

import numpy as np
import fasttext
import pyonmttok
import torch

from purano.annotator.processors import Processor
from purano.models import Document
from purano.proto.info_pb2 import Info as InfoPb
from purano.util import tokenize


@Processor.register("fasttext")
class FasttextProcessor(Processor):
    def __init__(
        self,
        vector_model_path: str,
        torch_model_path: str = None
    ):
        self.vector_model = fasttext.load_model(vector_model_path)
        self.torch_model = torch.load(torch_model_path) if torch_model_path else None
        self.tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=False)

    def __call__(
        self,
        docs: List[Document],
        infos: List[InfoPb],
        input_fields: List[str],
        output_field: str,
        max_tokens_count: int,
        agg_type: str,
        norm_word_vectors: bool = True,
        use_preprocessing: bool = True,
        **kwargs
    ):
        assert agg_type in ("mean", "max", "mean||max||min", "linear")
        assert agg_type != "linear" or self.torch_model
        assert len(kwargs) == 0, "{} are not used".format(kwargs)

        word_embeddings = self.calc_word_embeddings(
            docs, input_fields, max_tokens_count,
            norm_word_vectors=norm_word_vectors,
            use_preprocessing=use_preprocessing
        )

        final_embeddings = None
        mean_embeddings = np.mean(word_embeddings, axis=1)
        max_embeddings = np.max(word_embeddings, axis=1)
        min_embeddings = np.min(word_embeddings, axis=1)
        if agg_type == "mean":
            final_embeddings = mean_embeddings
        elif agg_type == "max":
            final_embeddings = max_embeddings
        elif agg_type == "mean||max||min" or agg_type == "linear":
            all_embeddings = (mean_embeddings, max_embeddings, min_embeddings)
            final_embeddings = np.concatenate(all_embeddings, axis=1)
            if agg_type == "linear":
                final_embeddings = self.torch_model(torch.tensor(final_embeddings))

        assert final_embeddings is not None
        for doc_num, info in enumerate(infos):
            getattr(info, output_field).extend(final_embeddings[doc_num])

    def calc_word_embeddings(
        self,
        docs: List[Document],
        input_fields: List[str],
        max_tokens_count: int,
        norm_word_vectors: bool,
        use_preprocessing: bool
    ):
        batch_size = len(docs)
        vector_dim = self.vector_model.get_dimension()
        word_embeddings = np.zeros((batch_size, max_tokens_count, vector_dim), dtype=np.float32)
        real_max_tokens_count = 0
        for doc_num, doc in enumerate(docs):
            sample = " ".join([getattr(doc, input_field) for input_field in input_fields])
            tokens = tokenize(sample) if use_preprocessing else sample.split(" ")
            tokens = tokens[:max_tokens_count]
            real_max_tokens_count = max(real_max_tokens_count, len(tokens))
            for token_num, token in enumerate(tokens):
                word_vector = self.vector_model.get_word_vector(token)
                if norm_word_vectors:
                    word_vector /= np.linalg.norm(word_vector)
                word_embeddings[doc_num, token_num, :] = word_vector
        word_embeddings = word_embeddings[:, :real_max_tokens_count, :]
        return word_embeddings
