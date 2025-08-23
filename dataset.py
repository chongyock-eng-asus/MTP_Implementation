import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset
from datasets import load_dataset

class MultiTokenPredictionDataset(Dataset):
    def __init__(self, ds, tokenizer, config):
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_length = config['max_length']

        self.num_mask = config['num_masks']
        self.pad_token_id = self.tokenizer.eos_token_id

        self.mask_tokens = [f"<mask_{i}>" for i in range(self.num_mask)]
        self.mask_token_ids = self.tokenizer.convert_tokens_to_ids(self.mask_tokens)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):

        input_sentence = self.ds[idx]
        tokenized_input = self.tokenizer.encode(input_sentence)
        input_id, position_id, labels, mtp_mask = self._create_masked_input(tokenized_input)
        return {
            'input_ids': input_id,
            'position_ids': position_id,
            'mtp_mask': mtp_mask,
            'labels':labels
        }

    def _create_masked_input(self, sequence):
        input_id = []
        position_id = []
        labels = []
        mtp_mask = []

        for i in range(len(sequence)):  
            input_id.append(sequence[i])
            input_id += self.mask_token_ids
            position_id.extend([i for i in range(i, i+self.num_mask+1)])
            labels.extend([
                sequence[i+1+j] if i+1+j < len(sequence) else -100 
                for j in range(self.num_mask+1)
            ])
            mtp_mask.extend([False]+self.num_mask*[True])

        # Truncate if too long
        if len(input_id) > self.max_length:
            input_id = input_id[:self.max_length]
            position_id = position_id[:self.max_length]
            labels = labels[:self.max_length]
            mtp_mask = mtp_mask[:self.max_length]
    
        # Pad if shorter than max_length
        pad_len = self.max_length - len(input_id)
        if pad_len > 0:
            input_id += [self.pad_token_id] * pad_len
            position_id += [0] * pad_len
            labels += [-100] * pad_len
            mtp_mask += [False] * pad_len

        input_id = torch.tensor(input_id, dtype=torch.long)
        position_id = torch.tensor(position_id, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        mtp_mask = torch.tensor(mtp_mask, dtype=torch.bool)
        
        return input_id, position_id, labels, mtp_mask
    
def get_ds(config):
    ds = load_dataset(config["datasource"], split=f"train[:{config['dataset_size']}]", token=config["API_KEY"])
    texts = []
    for example in ds:
          messages = example['messages']
          text_parts = [f"{msg['role']}: {msg['content']}" for msg in messages]
          full_text = "\n".join(text_parts)
          texts.append(full_text)
    return texts
