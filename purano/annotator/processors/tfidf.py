from typing import List, Optional

import numpy as np
import torch

from purano.annotator.processors import Processor
from purano.models import Document
from purano.proto.info_pb2 import Info as InfoPb
from purano.training.models.tfidf import load_idfs, get_tfidf_vector, SVDEmbedder


@Processor.register("tfidf")
class TfIdfProcessor(Processor):
    def __init__(
        self,
        idfs_vocabulary: str,
        svd_torch_model_path: str
    ):
        word2idf, word2idx = load_idfs(idfs_vocabulary)
        self.word2idf = word2idf
        self.word2idx = word2idx
        self.svd_torch_model = None  # type: Optional[SVDEmbedder]
        if svd_torch_model_path:
            self.svd_torch_model = torch.load(svd_torch_model_path)

    def __call__(
        self,
        docs: List[Document],
        infos: List[InfoPb],
        input_fields: List[str],
        output_field: str,
    ):
        embeddings = np.zeros((len(docs), len(self.word2idf)), dtype=np.float32)
        for doc_num, (doc, info) in enumerate(zip(docs, infos)):
            text = " ".join([getattr(doc, field) for field in input_fields])
            data, indices = get_tfidf_vector(text, self.word2idf, self.word2idx)
            for index, value in zip(indices, data):
                embeddings[doc_num][index] = value
        final_embeddings = embeddings
        if self.svd_torch_model:
            final_embeddings = self.svd_torch_model(torch.FloatTensor(final_embeddings))
        for doc_num, info in enumerate(infos):
            getattr(info, output_field).extend(final_embeddings[doc_num])
