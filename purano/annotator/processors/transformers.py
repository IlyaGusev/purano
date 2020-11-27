import os
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from purano.annotator.processors import Processor
from purano.models import Document
from purano.proto.info_pb2 import Info as InfoPb

@Processor.register("transformers")
class TransformersProcessor(Processor):
    def __init__(self, pretrained_model_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.model.eval()

    def __call__(self,
        docs: List[Document],
        infos: List[InfoPb],
        input_fields: List[str],
        output_field: str,
        max_tokens_count: int,
        layer: int,
        aggregation: str = "mean||max"
    ):
        batch_input_ids = torch.zeros((len(docs), max_tokens_count), dtype=int)
        batch_mask = torch.zeros((len(docs), max_tokens_count), dtype=int)
        for doc_num, doc in enumerate(docs):
            sample = " ".join([getattr(doc, input_field) for input_field in input_fields])
            inputs = self.tokenizer(sample,
                add_special_tokens=True,
                max_length=max_tokens_count,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            )
            input_ids = inputs["input_ids"][0]
            attention_mask = inputs["attention_mask"][0]
            assert len(input_ids) == len(attention_mask) == max_tokens_count
            batch_input_ids[doc_num, :len(input_ids)] = input_ids
            batch_mask[doc_num, :len(attention_mask)] = attention_mask
        with torch.no_grad():
            output = self.model(batch_input_ids, attention_mask=batch_mask, return_dict=True, output_hidden_states=True)
        layer_embeddings = output.hidden_states[layer]
        embeddings = layer_embeddings.cpu().numpy()
        if aggregation == "mean||max":
            embeddings = np.concatenate((embeddings.mean(axis=1), embeddings.max(axis=1)), axis=1)
        elif aggregation ==  "first":
            embeddings = embeddings[:, 0, :]
        else:
            assert False
        for doc_num, info in enumerate(infos):
            getattr(info, output_field).extend(embeddings[doc_num])
