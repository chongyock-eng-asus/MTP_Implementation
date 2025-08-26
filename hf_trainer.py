import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Tuple, Dict, List,  Union
from train import train_mtp_model
from dataset import MultiTokenPredictionDataset, get_ds
from model import MultiTokenPredictionModel
from utils import get_tokenizer, test_hypothesis
from config import get_config

class MultiTokenPredictionConfig(PretrainedConfig):
    model_type = "multi_token_prediction"
    
    def __init__(
        self,
        model_basename: str = "meta-llama/Llama-3.2-1B",
        num_masks: int = 4,
        lora_rank: int = 128,
        lora_alpha: int = 256,
        API_KEY: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_basename = model_basename
        self.num_masks = num_masks
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.API_KEY = API_KEY
        
class MultiTokenPredictionDataset(Dataset):
    def __init__(self, ds, tokenizer, max_length=256, num_masks=4):
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.num_mask = num_masks
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

class HFMultiTokenPredictionModel(PreTrainedModel):
    config_class = MultiTokenPredictionConfig
    
    def __init__(self, config: MultiTokenPredictionConfig, tokenizer=None):
        super().__init__(config)
        self.config = config
        
        # Convert config to dictionary format that your original model expects
        model_config = {
            'model_basename': config.model_basename,
            'num_masks': config.num_masks,
            'lora_rank': config.lora_rank,
            'lora_alpha': config.lora_alpha,
            'API_KEY': config.API_KEY
        }
        
        # Initialize your original model
        self.mtp_model = MultiTokenPredictionModel(model_config, tokenizer)
        
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mtp_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Call your original model's forward method
        outputs = self.mtp_model.forward(
            input_ids=input_ids,
            mtp_mask=mtp_mask,
            labels=labels,
            position_ids=position_ids
        )
        
        # Extract components
        logits = outputs['logits']
        loss = outputs.get('total_loss', None) if labels is not None else None
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    
    def generate(self, text: str):
        """Wrapper for your generate method"""
        return self.mtp_model.generate(text)
    
    def get_trainable_parameters(self):
        """Get trainable parameters"""
        return self.mtp_model.get_trainable_parameters()

class MultiTokenPredictionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract your custom inputs
        input_ids = inputs.get("input_ids")
        mtp_mask = inputs.get("mtp_mask")
        position_ids = inputs.get("position_ids")
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            mtp_mask=mtp_mask,
            position_ids=position_ids,
            labels=labels
        )
        
        # Get loss from outputs
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        
        return (loss, outputs) if return_outputs else loss

def setup_model_and_tokenizer(config_dict, tokenizer_name_or_path):
    """Setup your model with HuggingFace compatibility"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, token=config_dict["API_KEY"])
    
    # Add special tokens if needed (adapt this to your tokenizer setup)
    if not hasattr(tokenizer, 'custom_mask_token_ids'):
        # Add your custom mask tokens
        special_tokens = [f"<mask_{i}>" for i in range(config_dict['num_masks'])]
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        tokenizer.custom_mask_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in special_tokens]
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create config
    config = MultiTokenPredictionConfig(**config_dict)
    
    # Create model
    model = HFMultiTokenPredictionModel(config, tokenizer)
    
    return model, tokenizer, config

def train_model(model, tokenizer, train_data, eval_data=None, output_dir="./mtp_model_output"):
    """Train your model using HuggingFace Trainer"""
    
    # Create datasets
    train_dataset = MultiTokenPredictionDataset(train_data, tokenizer, num_masks=model.config.num_masks)
    eval_dataset = MultiTokenPredictionDataset(eval_data, tokenizer, num_masks=model.config.num_masks) if eval_data else None
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Adjust based on your memory
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=200 if eval_dataset else None,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues
        remove_unused_columns=False,  # Important for custom inputs
    )
    
    # Create trainer
    trainer = MultiTokenPredictionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    return trainer

if __name__ == "__main__":
    config_dict = get_config()
    ds = get_ds(config_dict)
    model, tokenizer, config = setup_model_and_tokenizer(config_dict, tokenizer_name_or_path=config_dict['model_basename'])
    trainer = train_model(model, tokenizer, ds)
    print("Starting training...")
    trainer.train()
    trainer.save_model("./final_mtp_model")
    tokenizer.save_pretrained("./final_mtp_model")
    print("Training completed and model saved!")