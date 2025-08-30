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
import argparse

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
            position_ids=position_ids,
            attention_bias=attention_bias,
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
        attention_bias = inputs.get("attention_bias")
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            mtp_mask=mtp_mask,
            position_ids=position_ids,
            labels=labels,
            attention_mask=attention_bias
        )
        
        # Get loss from outputs
        loss = outputs.loss
                
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

def train_model(model, tokenizer, train_data, config, eval_data=None, output_dir="./mtp_model_output"):
    """Train your model using HuggingFace Trainer"""
    
    # Create datasets
    train_dataset = MultiTokenPredictionDataset(train_data, tokenizer, config)
    eval_dataset = MultiTokenPredictionDataset(eval_data, tokenizer, config) if eval_data else None
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,  # Adjust based on your memory
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=200 if eval_dataset else None,
        save_steps=10000,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues
        remove_unused_columns=False,  # Important for custom inputs
        save_safetensors=False,
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True, help="Hugging Face API key")
    args = parser.parse_args()
    config_dict = get_config()
    config_dict["API_KEY"] = args.api_key
    
    ds = get_ds(config_dict)
    model, tokenizer, config = setup_model_and_tokenizer(config_dict, tokenizer_name_or_path=config_dict['model_basename'])
    trainer = train_model(model, tokenizer, ds)
    print("Starting training...")
    trainer.train()
    trainer.save_model("./final_mtp_model")
    tokenizer.save_pretrained("./final_mtp_model")
    print("Training completed and model saved!")
