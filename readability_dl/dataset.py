from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch

class CLRPDataset(Dataset):
    def __init__(self, data, checkpoint, max_length: int = 128, is_test: bool = False):
        self.excerpts = data.excerpt.values.tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.max_len = max_length
        self.targets = data.target.values.tolist()
        
    def __getitem__(self, idx):
        item = self.tokenizer(self.excerpts[idx], max_length=self.max_len,
                             return_tensors="pt", truncation=True, padding="max_length")

        if self.is_test:
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "token_type_ids": torch.tensor(item["token_type_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long)
            }
        else:
            target = self.targets[idx]
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "token_type_ids": torch.tensor(item["token_type_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "label": torch.tensor(target, dtype=torch.double)
            }
            
    def __len__(self):
        return len(self.targets)