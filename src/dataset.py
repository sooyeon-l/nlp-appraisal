import torch
import pandas as pd
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

class AppraisalDataset(Dataset): 
    def __init__(self, csv_path: str, tokenizer, weights: dict, target_dims: list): 
        weights = {dim_name: {int(rating): weight for rating, weight in dim_weights.items()} for dim_name, dim_weights in weights.items()}
        self.target_dims = target_dims
        df = pd.read_csv(csv_path) 
        # De-normalize ratings first and then compute the sample weights
        self.sample_weights = torch.tensor(
            pd.DataFrame({
                dim: (df[dim] * 4 + 1).round().astype(int).map(weights[dim]) for dim in target_dims if dim in df.columns
            }).values, dtype=torch.float32
        )
        hf_dataset = HFDataset.from_pandas(df)
        # Tokenize everything upfront and cache it
        def tokenize_fn(examples):
            return tokenizer(examples['generated_text'], truncation=True, padding='max_length', max_length=128)
        self.tokenized_dataset = hf_dataset.map(tokenize_fn, batched=True)

    def __len__(self) -> int:
        return len(self.tokenized_dataset)
    
    def __getitem__(self, idx:int):
        item = self.tokenized_dataset[idx]
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
        labels = torch.tensor([item[dim] for dim in self.target_dims], dtype=torch.float32)
        weights = self.sample_weights[idx]
        
        return {            
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,       
                'weights': weights
        }