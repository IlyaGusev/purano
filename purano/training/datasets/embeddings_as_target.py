from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class EmbeddingsAsTargetDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        url2num: Dict[str, int],
        num2embedding: Dict[int, torch.tensor],
        tokenizer: AutoTokenizer,
        max_tokens_count: int = 196
    ):
        self.samples = []
        for record in records:
            num = url2num.pop(record["url"])
            embedding = num2embedding[num]
            text = record["text"]
            inputs = tokenizer(
                text,
                add_special_tokens=True,
                max_length=max_tokens_count,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            )
            self.samples.append({
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "embedding": embedding
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
