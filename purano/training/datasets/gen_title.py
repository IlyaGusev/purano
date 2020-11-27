import torch
from torch.utils.data import Dataset


class GenTitleDataset(Dataset):
    def __init__(
        self,
        records,
        tokenizer,
        max_tokens_text=200,
        max_tokens_title=40
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_tokens_text = max_tokens_text
        self.max_tokens_title = max_tokens_title

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        title = record.get("title", "")
        text = record["text"]
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_tokens_text,
            padding="max_length",
            truncation=True)
        outputs = self.tokenizer(
            title,
            add_special_tokens=True,
            max_length=self.max_tokens_title,
            padding="max_length",
            truncation=True)
        decoder_input_ids = torch.tensor(outputs["input_ids"])
        decoder_attention_mask = torch.tensor(outputs["attention_mask"])
        labels = decoder_input_ids.clone()
        for i, mask in enumerate(decoder_attention_mask):
            if mask == 0:
                labels[i] = -100
        return {
            "input_ids": torch.tensor(inputs["input_ids"]),
            "attention_mask": torch.tensor(inputs["attention_mask"]),
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels
        }
