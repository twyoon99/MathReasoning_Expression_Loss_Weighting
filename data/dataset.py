import json
import torch
from torch.utils.data import Dataset

class BaseMathDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.samples = []

        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                question = data["question"].strip()
                answer = data["answer"].strip()

                prompt = f"### Question:\n{question}\n\n### Answer:\n"
                full_text = prompt + answer

                encoded = self.tokenizer(
                    full_text,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )

                input_ids = encoded["input_ids"][0]
                attention_mask = encoded["attention_mask"][0]
                labels = input_ids.clone()

                # question 마스킹 -100
                prompt_len = len(self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]) - 1 
                labels[:prompt_len] = -100

                self.samples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
