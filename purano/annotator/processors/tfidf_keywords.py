from typing import List

from purano.annotator.processors import Processor
from purano.models import Document
from purano.proto.info_pb2 import Info as InfoPb
from purano.training.models.tfidf import load_idfs, get_tfidf_vector


@Processor.register("tfidf_keywords")
class TfIdfKeywordsProcessor(Processor):
    def __init__(
        self,
        idfs_vocabulary: str,
        top_k: int
    ):
        self.top_k = top_k
        word2idf, word2idx = load_idfs(idfs_vocabulary)
        self.word2idf = word2idf
        self.word2idx = word2idx
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def __call__(
        self,
        docs: List[Document],
        infos: List[InfoPb],
        input_fields: List[str],
        output_field: str,
    ):
        for doc, info in zip(docs, infos):
            sample = " ".join([getattr(doc, field) for field in input_fields])
            data, indices = get_tfidf_vector(sample, self.word2idf, self.word2idx)
            tfidfs = [(tfidf, self.idx2word[index]) for tfidf, index in zip(data, indices)]
            tfidfs.sort()
            keywords = [token for tfidf, token in tfidfs[-self.top_k:]]
            getattr(info, output_field).extend(keywords)
