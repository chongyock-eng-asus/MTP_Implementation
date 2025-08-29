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
        input_sentence = self.ds['conversations'][idx]
        messages = self.convert_format(input_sentence)
        
        tokenized_input = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).squeeze(0)

        input_id, position_id, labels, mtp_mask = self._create_masked_input(tokenized_input)
        return {
            'input_ids': input_id,
            'position_ids': position_id,
            'mtp_mask': mtp_mask,
            'labels':labels
        }
    
    @staticmethod
    def convert_format(conversations):
        """Convert from dataset format to chat template format"""
        converted = []
        for msg in conversations:
            role = "user" if msg['from'] == 'human' else "assistant"
            converted.append({
                "role": role,
                "content": msg['value']
            })
        return converted

    def _create_masked_input(self, sequence):
        input_id = []
        position_id = []
        labels = []
        mtp_mask = []

        for i in range(len(sequence)):
            # Add original token
            input_id.append(sequence[i])

            # Add mask tokens
            input_id += self.mask_token_ids

            # Position IDs: original token gets position i, masks get i+1, i+2, etc.
            position_id.extend([i] + [i+1+j for j in range(self.num_mask)])

            # Labels: original token predicts next token, masks predict future tokens
            labels.append(sequence[i+1] if i+1 < len(sequence) else -100)  # Original token label

            # Mask token labels: each mask predicts a future token
            for j in range(self.num_mask):
                future_idx = i + 1 + j + 1
                if future_idx < len(sequence):
                    labels.append(sequence[future_idx])
                else:
                    labels.append(-100)

            # MTP mask: original token = False, mask tokens = True
            mtp_mask.extend([False] + [True] * self.num_mask)

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
    # Load the dataset
    ds = load_dataset(config["datasource"], split=f"train[:{config['dataset_size']}]", token=config["API_KEY"])
    return ds
