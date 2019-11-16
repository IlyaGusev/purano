import requests
import json
from typing import List

from razdel import sentenize
from purano.proto.info_pb2 import Info as InfoPb, EntitySpan as EntitySpanPb


class NerClient:
    def __init__(self,
                 ip: str,
                 port: int,
                 max_char_count: int=1000,
                 sentence_batch_size: int=128):
        self.ip = ip
        self.port = port
        self.max_char_count = max_char_count
        self.sentence_batch_size = sentence_batch_size

    def encode(self, docs: List[str]) -> List[List[EntitySpanPb]]:
        sentences = [(i, s) for i, doc in enumerate(docs) for s in sentenize(doc)]
        texts_only = [s.text[:self.max_char_count] for _, s in sentences]
        answers = []
        batch_begin = 0
        while batch_begin < len(texts_only):
            batch_end = batch_begin + self.sentence_batch_size
            batch = texts_only[batch_begin:batch_end]
            data = json.dumps({"x": batch})
            url = "http://{ip}:{port}/model".format(ip=self.ip, port=self.port)
            r = requests.post(url, data=data)
            answers += r.json()
            batch_begin += self.sentence_batch_size
        outputs = []
        spans = []
        current_doc_num = 0
        for (doc_num, sentence), (tokens, tags) in zip(sentences, answers):
            if current_doc_num != doc_num:
                outputs.append(spans)
                spans = []
                current_doc_num = doc_num
            sentence_spans = self.get_spans(sentence.text, tokens, tags)
            for span in sentence_spans:
                span.begin += sentence.start
                span.end += sentence.start
            spans += sentence_spans
        outputs.append(spans)
        return outputs

    @classmethod
    def get_spans(cls, text: str, tokens: List[str],
                  tags: List[str]) -> List[EntitySpanPb]:
        begin = 0
        spans = []
        for token, tag in zip(tokens, tags):
            token_begin = text.find(token, begin)
            token_end = token_begin + len(token)
            begin = token_end
            if tag.startswith("B"):
                entity_span = EntitySpanPb()
                entity_span.begin = token_begin
                entity_span.end = token_end
                tag_name = tag.split("-")[-1].strip()
                entity_span.tag = EntitySpanPb.Tag.Value(tag_name)
                spans.append(entity_span)
            elif tag.startswith("I") and spans:
                spans[-1].end = token_end
        return spans


