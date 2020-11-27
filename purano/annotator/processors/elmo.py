from typing import List

import numpy as np
import torch
import pyonmttok

from purano.annotator.processors import Processor
from purano.models import Document
from purano.proto.info_pb2 import Info as InfoPb


@Processor.register("elmo")
class ElmoProcessor(Processor):
    def __init__(self, options_file: str, weight_file: str, cuda_device: int):
        from allennlp.modules.elmo import _ElmoBiLm
        from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
        self.indexer = ELMoTokenCharactersIndexer()
        self.elmo_bilm = _ElmoBiLm(options_file, weight_file)
        if cuda_device >= 0:
            self.elmo_bilm = self.elmo_bilm.cuda(device=cuda_device)
        self.cuda_device = cuda_device
        self.tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=False)

    def __call__(
        self,
        docs: List[Document],
        infos: List[InfoPb],
        input_fields: List[str],
        output_field: str,
        max_tokens_count: int
    ):
        from allennlp.modules.elmo import batch_to_ids
        from allennlp.nn.util import remove_sentence_boundaries
        batch = []
        for doc_num, doc in enumerate(docs):
            sample = " ".join([getattr(doc, input_field) for input_field in input_fields])
            tokens = self.preprocess(sample)[:max_tokens_count]
            batch.append(tokens)
        character_ids = batch_to_ids(batch)
        if self.cuda_device >= 0:
            character_ids = character_ids.cuda(device=self.cuda_device)
        bilm_output = self.elmo_bilm(character_ids)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']
        without_bos_eos = [
            remove_sentence_boundaries(layer, mask_with_bos_eos)
            for layer in layer_activations
        ]
        embeddings = torch.cat([pair[0].unsqueeze(1) for pair in without_bos_eos], dim=1)
        mask = without_bos_eos[0][1]
        for doc_num, info in enumerate(infos):
            length = int(mask[doc_num, :].sum())
            doc_embeddings = np.zeros((3, 0, 1024))
            if length != 0:
                doc_embeddings = embeddings[doc_num, :, :length, :].detach().cpu().numpy()
            doc_embeddings = doc_embeddings.swapaxes(0, 1).reshape(doc_embeddings.shape[0], -1)
            mean_embeddings = doc_embeddings.mean(axis=0)
            max_embeddings = doc_embeddings.max(axis=0)
            final_embedding = np.concatenate((mean_embeddings, max_embeddings), axis=0)
            getattr(info, output_field).extend(final_embedding)

    def preprocess(self, text):
        text = str(text).strip().replace("\n", " ").replace("\xa0", " ")
        tokens, _ = self.tokenizer.tokenize(text)
        return tokens
