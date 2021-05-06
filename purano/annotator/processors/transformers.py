from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, logging

from purano.annotator.processors import Processor
from purano.models import Document
from purano.proto.info_pb2 import Info as InfoPb


@Processor.register("transformers")
class TransformersProcessor(Processor):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        log_info: bool = False,
        use_gpu: bool = False,
        do_lower_case: bool = False,
        do_basic_tokenize: bool = True,
        strip_accents: bool = True
    ):
        if log_info:
            logging.set_verbosity_info()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            strip_accents=strip_accents
        )
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
        self.model.eval()

    def __call__(
        self,
        docs: List[Document],
        infos: List[InfoPb],
        input_fields: List[str],
        output_field: str,
        max_tokens_count: int,
        layer: int,
        aggregation: str = "mean||max"
    ):
        samples = []
        for doc_num, doc in enumerate(docs):
            samples.append(" ".join([getattr(doc, input_field) for input_field in input_fields]))
        inputs = self.tokenizer(
            samples,
            add_special_tokens=True,
            max_length=max_tokens_count,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        batch_input_ids = inputs["input_ids"]
        batch_mask = inputs["attention_mask"]
        assert len(batch_input_ids[0]) == len(batch_mask[0]) == max_tokens_count
        if self.use_gpu:
            batch_input_ids = batch_input_ids.cuda()
            batch_mask = batch_mask.cuda()
        with torch.no_grad():
            output = self.model(
                batch_input_ids,
                attention_mask=batch_mask,
                return_dict=True,
                output_hidden_states=True
            )
        layer_embeddings = output.hidden_states[layer]
        embeddings = layer_embeddings.cpu().numpy()
        if aggregation == "mean||max":
            embeddings = np.concatenate((embeddings.mean(axis=1), embeddings.max(axis=1)), axis=1)
        elif aggregation == "first":
            embeddings = embeddings[:, 0, :]
        else:
            assert False
        for doc_num, info in enumerate(infos):
            getattr(info, output_field).extend(embeddings[doc_num])
