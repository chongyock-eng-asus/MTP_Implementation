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
    
    # Load Llama 3.2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    # Set the correct Llama 3.1/3.2 chat template (includes JSON tool calling support)
    tokenizer.chat_template = "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + tool definitions #}\n{%- if tools is not none %}\n    {%- set system_message = \"Environment: ipython\\nToday Date: \" + date_string + \"\\n\\nYou have access to the following functions. To call a function, please respond with JSON for the function call only.\\nCalling any function is optional. If you call a function, the result will be shown to you.\\nIf the function result indicates error, please fix the error and call the function again. Do not pretend to know the result of the function call.\\n\\n\" + tools|string %}\n{%- endif %}\n\n{%- if system_message is defined and system_message!=\"\" %}\n    {{- '<|start_header_id|>system<|end_header_id|>\\n\\n' }}\n    {{- system_message }}\n    {{- '<|eot_id|>' }}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if message['role'] == 'user' or message['role'] == 'system' %}\n        {%- if tools is not none and (message == messages[-1]) %}\n            {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n            {%- if tools_in_user_message %}\n                {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n                {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n                {{- tools|string }}\n                {{- \"\\n\\nQuestion: \" + message['content'] + '<|eot_id|>' }}\n            {%- else %}\n                {{- message['content'] + '<|eot_id|>' }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n        {%- if message['content'] is not none %}\n            {{- message['content'] | trim }}\n        {%- endif %}\n        {%- if 'tool_calls' in message and message['tool_calls'] is not none %}\n            {%- for tool_call in message['tool_calls'] %}\n                {{- json.dumps(tool_call) }}\n                {{- \"\\n\" }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|eot_id|>' }}\n    {%- elif message['role'] == 'tool' %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1]['role'] != 'tool') %}\n            {{- '<|start_header_id|>ipython<|end_header_id|>\\n\\n' }}\n        {%- endif %}\n        {%- if message['content'] is not none and message['content']|length > 0 %}\n            {{- message['content'] | trim }}\n        {%- endif %}\n        {%- if loop.last or (messages[loop.index0 + 1]['role'] != 'tool') %}\n            {{- '<|eot_id|>' }}\n        {%- else %}\n            {{- '\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}"
    
    texts = []
    for example in ds:
        messages = example['messages']
        
        # Use Llama's chat template - this handles all special tokens correctly
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,  # Return string, not tokens
            add_generation_prompt=False  # Don't add prompt for generation since we have complete conversations
        )
        
        texts.append(formatted_text)
    
    return texts
