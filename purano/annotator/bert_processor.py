import os
from typing import List

import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, BertForPreTraining

class BertProcessor:
    def __init__(self,
                 pretrained_model_name_or_path: str,
                 max_tokens_count: int=64,
                 config_file_name: str="bert_config.json",
                 model_ckpt_file_name: str="bert_model.ckpt.index",
                 layer: int=-2):
        config_full_path = os.path.join(pretrained_model_name_or_path, config_file_name)
        ckpt_full_path = os.path.join(pretrained_model_name_or_path, model_ckpt_file_name)

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        config = BertConfig.from_json_file(config_full_path)
        config.output_hidden_states = True
        self.model = BertForPreTraining.from_pretrained(ckpt_full_path, from_tf=True, config=config).bert
        self.max_tokens_count = max_tokens_count
        self.layer = layer

    def encode(self, docs: List[str]) -> np.array:
        batch_input_ids = torch.zeros((len(docs), self.max_tokens_count), dtype=int)
        batch_mask = torch.zeros((len(docs), self.max_tokens_count), dtype=int)
        for i, sample in enumerate(docs):
            tokens = self.tokenizer.tokenize(sample)
            tokens = ['[CLS]'] + tokens[:self.max_tokens_count-2] + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            batch_input_ids[i, :len(input_ids)] = torch.tensor(input_ids)
            batch_mask[i, :len(input_ids)] = torch.ones((len(input_ids), ), dtype=int)

        self.model.eval()
        with torch.no_grad():
            output = self.model(batch_input_ids, attention_mask=batch_mask)[-1]
        output = output[self.layer]
        embeddings = output.cpu().numpy()
        embeddings = np.concatenate((embeddings.mean(axis=1), embeddings.max(axis=1)), axis=1)
        return embeddings

