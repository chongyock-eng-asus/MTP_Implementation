import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, Dict, List
import math
import time

class GatedLoRALinear(nn.Module):
    """
    Gated LoRA layer that only applies adaptations to MTP tokens
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 128, alpha: int = 256, dropout: float = 0.1):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA parameters
        device = base_layer.weight.device 
        self.lora_A = nn.Parameter(torch.randn(rank, base_layer.in_features, device=device) * 0.01) # [rank, in_features]
        self.lora_B = nn.Parameter(torch.randn(base_layer.out_features, rank, device=device) * 0.01) # [out_features, rank]
        self.dropout = nn.Dropout(dropout)

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base transformation
        base_output = self.base_layer(x)

        return base_output


class SamplerHead(nn.Module):
    """
    Lightweight sampler head for coherent sequence generation
    """
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Two-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size)
        )

        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor, previous_embeddings: torch.Tensor) -> torch.Tensor:
        # Concatenate hidden states with previous token embeddings
        combined = torch.cat([hidden_states, previous_embeddings], dim=-1)

        # Pass through MLP
        features = self.mlp(combined)

        # Project to vocabulary
        logits = self.output_projection(features)

        return logits


class MultiTokenPredictionModel(nn.Module):
    """
    Multi-Token Prediction Model with Gated LoRA and Sampler Head
    """
    def __init__(self, config, tokenizer, device=None):
        super().__init__()

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(config['model_basename'], token=config['API_KEY'], device_map='auto')
        self.config = self.base_model.config
        self.num_masks = config['num_masks']
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)

        self._apply_gated_lora(config['lora_rank'], config['lora_alpha'])

        # Resize embeddings
        self.base_model.resize_token_embeddings(self.vocab_size)

        # Create sampler head
        self.sampler_head = SamplerHead(
            self.config.hidden_size,
            len(self.tokenizer)  # Account for new mask tokens
        )

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Keep LoRA and sampler parameters trainable
        for name, module in self.named_modules():
            if isinstance(module, GatedLoRALinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True

        for param in self.sampler_head.parameters():
            param.requires_grad = True

        # Hook portion to review
        self.current_mtp_mask = None
        self._register_permanent_hooks()

        if device is not None:
            self.base_model = self.base_model.to(device)
            self.sampler_head = self.sampler_head.to(device)

    def _gated_lora_hook(self, module, input, output):
        """
        Permanent hook function - gets called for EVERY forward pass
        The current_mtp_mask changes for each batch!
        """
        x = input[0]
        base_output = output
        
        if self.current_mtp_mask is None:
            return base_output

        lora_output = module.dropout(x) @ module.lora_A.T @ module.lora_B.T * module.scaling
        mask_expanded = self.current_mtp_mask.unsqueeze(-1)
        gated_output = torch.where(mask_expanded, base_output + lora_output, base_output)
        return gated_output

    def _register_permanent_hooks(self):
        """Register hooks once during initialization"""
        for module in self.base_model.modules():
            if isinstance(module, GatedLoRALinear):
                module.register_forward_hook(self._gated_lora_hook)

    def _apply_gated_lora(self, rank: int, alpha: int):
        """Apply Gated LoRA to transformer layers"""
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
                # Replace with Gated LoRA
                gated_layer = GatedLoRALinear(module, rank, alpha)

                # Set the layer in the model
                parent_name = '.'.join(name.split('.')[:-1])
                layer_name = name.split('.')[-1]
                parent_module = self.base_model.get_submodule(parent_name)
                setattr(parent_module, layer_name, gated_layer)


    def forward(
        self,
        input_ids: torch.Tensor,
        mtp_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # Introduce mtp_mask hooks here
        self.current_mtp_mask = mtp_mask
        num_heads = self.base_model.config.num_attention_heads
        # Forward pass through base model
        outputs = self.base_model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            attention_mask=.unsqueeze(1).expand(-1, num_heads, -1, -1),
        )
        logits = outputs.logits
        hidden_state = outputs.hidden_states[-1]

        base_loss = self._calculate_base_loss(logits, labels)
        sampler_loss = self._calculate_sampler_loss(input_ids, hidden_state, mtp_mask, labels)
        lcm_loss = self._calculate_lcm_loss(position_ids, hidden_state, mtp_mask)
        total_loss = base_loss + sampler_loss + lcm_loss
        self.current_mtp_mask = None

        return {
            'logits':logits,
            'total_loss':total_loss,
            'base_loss':base_loss,
            'sampler_loss':sampler_loss,
            'lcm_loss':lcm_loss,
            }

    def _calculate_base_loss(self, logits: torch.Tensor, labels:torch.Tensor):
        base_loss = F.cross_entropy(
        logits.view(-1, self.vocab_size), # [batch * seq_len, vocab_size]
        labels.view(-1), # [seq_len]
        ignore_index=-100)
        return base_loss


    def _calculate_sampler_loss(self, input_ids: torch.Tensor, hidden_state: torch.Tensor, mtp_mask: torch.Tensor, labels: torch.Tensor):
        
        embedding = self.base_model.get_input_embeddings()
        
        # Shift everything by 1 position
        # We want to predict position i using information from position i-1
        prev_tokens = input_ids[:, :-1]           # tokens at positions [0, 1, 2, ..., seq_len-2]
        current_hidden = hidden_state[:, 1:]      # hidden states at positions [1, 2, 3, ..., seq_len-1]  
        current_targets = labels[:, 1:]           # targets at positions [1, 2, 3, ..., seq_len-1]
        mtp_positions = mtp_mask[:, 1:]           # MTP mask at positions [1, 2, 3, ..., seq_len-1]
        
        # Filter to only MTP positions
        if mtp_positions.sum() == 0:
            return torch.tensor(0.0, device=hidden_state.device, requires_grad=True)
        
        # Get valid MTP positions
        valid_prev_tokens = prev_tokens[mtp_positions]      # Previous tokens for MTP positions
        valid_current_hidden = current_hidden[mtp_positions] # Current hidden states for MTP positions  
        valid_targets = current_targets[mtp_positions]       # Current targets for MTP positions
        
        # Get embeddings of previous tokens
        prev_embeddings = embedding(valid_prev_tokens)
        
        # Sampler prediction: prev_token_embedding + current_hidden -> current_token
        sampler_logits = self.sampler_head(valid_current_hidden, prev_embeddings)
        
        # Cross entropy loss
        sampler_loss = F.cross_entropy(sampler_logits, valid_targets, ignore_index=-100)
    
        return sampler_loss

    def _calculate_lcm_loss(self, position_mask: torch.Tensor, hidden_state: torch.Tensor, mtp_mask: torch.Tensor):
        # Shapes
        B, L, H = hidden_state.shape
        
        # Masked tokens are invalid: we invert mask for valid ones
        valid_mask = ~mtp_mask  # shape: (B, L)
    
        # Expand position IDs and masks
        pos_i = position_mask.unsqueeze(2)  # (B, L, 1)
        pos_j = position_mask.unsqueeze(1)  # (B, 1, L)
    
        # Match positions: True where position IDs match
        same_position = (pos_i == pos_j)  # shape: (B, L, L)
    
        # Only compare where both i and j are valid (not masked)
        valid_i = valid_mask.unsqueeze(2)  # (B, L, 1)
        valid_j = valid_mask.unsqueeze(1)  # (B, 1, L)
        valid_pairs = valid_i & valid_j  # (B, L, L)
    
        # Only consider positions before current position (i > j)
        idxs = torch.arange(L, device=hidden_state.device)
        before_mask = idxs.view(1, 1, L) < idxs.view(1, L, 1)  # (1, L, L)
    
        # Combine all masks
        final_mask = same_position & valid_pairs & before_mask  # (B, L, L)
    
        # Expand hidden states for i and j comparisons
        hi = hidden_state.unsqueeze(2).expand(-1, L, L, -1)  # (B, L, L, H)
        hi = hi.detach()
        hj = hidden_state.unsqueeze(1).expand(-1, L, L, -1)  # (B, L, L, H)
    
        # Compute squared difference
        mse = ((hi - hj) ** 2).mean(dim=-1)  # (B, L, L)
    
        # Mask out invalid positions
        masked_mse = mse * final_mask  # (B, L, L)
    
        # Sum and normalize
        total_loss = masked_mse.sum()
        num_valid = final_mask.sum().clamp(min=1)  # avoid divide-by-zero

        return total_loss / num_valid

    def get_trainable_parameters(self):
        """Get all trainable parameters"""
        trainable_params = []

        # Get ALL trainable parameters directly
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)

        return trainable_params

    def generate(self, text: str):

        # For now generate autoregressively until all masks are filled
        tokenized_output = self.tokenizer(text)
        encoded_input = tokenized_output['input_ids']
        input_tensor = torch.tensor(encoded_input).unsqueeze(0).to(self.base_model.device)

        start_time = time.time()
        sc_output =  self.speculative_decoding(input_tensor)
        sc_inference_time = time.time() - start_time

        start_time = time.time()
        ar_output = self.autoregressive_decoding(input_tensor, num_steps=self.num_masks)
        ar_inference_time = time.time() - start_time

        sc_decoded_output = self.tokenizer.decode(sc_output.squeeze(0))
        ar_decoded_output = self.tokenizer.decode(ar_output.squeeze(0))


        return {
            'sc_output': sc_output,
            'sc_inference_time': sc_inference_time,
            'sc_decoded_output': sc_decoded_output,
            'ar_output': ar_output,
            'ar_inference_time': ar_inference_time,
            'ar_decoded_output':ar_decoded_output
            }



    def autoregressive_decoding(self, input_tensor:torch.tensor, num_steps:int):
        self.current_mtp_mask = None
        with torch.no_grad():
            for _ in range(num_steps):
                outputs = self.base_model(
                    input_ids=input_tensor,
                    output_hidden_states=False,
                    use_cache=False
                )
                next_token = torch.argmax(outputs.logits, dim=-1)[:,-1].unsqueeze(0)
                input_tensor = torch.concat([input_tensor, next_token], dim=-1)
        return input_tensor

    def speculative_decoding(self, input_tensor:torch.tensor):

        device = input_tensor.device
        masked_token_ids = self.tokenizer.custom_mask_token_ids
        masked_token_ids = self.tokenizer.custom_mask_token_ids
        masked_token_tensor = torch.tensor(masked_token_ids).unsqueeze(0).to(device)

        input_with_mask_tensor = torch.concat([input_tensor, masked_token_tensor], dim=-1)
        mtp_mask = torch.concat([
                    torch.zeros(input_tensor.shape[0], input_tensor.shape[1], device=device),
                    torch.ones(masked_token_tensor.shape[0], masked_token_tensor.shape[1], device=device)
                ], dim=-1).bool()

        self.current_mtp_mask = mtp_mask

        with torch.no_grad():
              outputs = self.base_model(
                      input_ids=input_with_mask_tensor,
                      output_hidden_states=True,
                      use_cache=False
                  )
              hidden_states = outputs.hidden_states[-1]
              current_ntp_idx = input_tensor.shape[1] - 1
              output_tokens = torch.argmax(outputs.logits, dim=-1)
              prev_token = output_tokens[:, current_ntp_idx].unsqueeze(0)
              input_with_mask_tensor[:, current_ntp_idx+1] = prev_token.squeeze(1)
              current_mtp_idx = current_ntp_idx + 1
              embedding = self.base_model.get_input_embeddings()
              for i in range(self.num_masks-1):

                  mtp_hidden = hidden_states[:, current_mtp_idx, :]

                  # Use the last token in prev_token for embedding
                  last_token = prev_token[:, -1]  # Shape: [batch_size]
                  prev_emb = embedding(last_token)  # Shape: [batch_size, embed_dim]

                  # Use sampler head for prediction
                  sampler_logits = self.sampler_head(mtp_hidden, prev_emb)
                  next_token = torch.argmax(sampler_logits, dim=-1, keepdim=True)  # Shape: [batch_size, 1]

                  # Update the mask tensor
                  input_with_mask_tensor[:, current_mtp_idx+1] = next_token.squeeze(1)

                  # Append the new token to prev_token (keep accumulating)
                  prev_token = torch.cat([prev_token, next_token], dim=1)  # Shape: [batch_size, seq_len + i + 1]

                  # Update for next iteration
                  current_mtp_idx += 1

        with torch.no_grad():
                outputs = self.base_model(
                            input_ids=input_with_mask_tensor,
                            output_hidden_states=False,
                            use_cache=False
                        )
                next_token = torch.argmax(outputs.logits, dim=-1)
                # Ignore the last token and check if :-(model.num_mask-1) matches
                ntp_verification_tensor = next_token[:,:-1]
                verification_tensor1 = ntp_verification_tensor[:, -(self.num_masks-1):]
                verification_tensor2 = input_with_mask_tensor[:,1:][:, -(self.num_masks-1):]
                mismatch_mask = verification_tensor1 != verification_tensor2
                print(f"Speculated tokens: {verification_tensor1}")
                print(f"Verification tokens: {verification_tensor2}")
                print(f"Mismatch mask:  {mismatch_mask}")

                if mismatch_mask.any():
                    first_mismatch_abs = torch.where(mismatch_mask)[1][0] + verification_tensor1.shape[1] - (self.num_masks - 1)
                    print(f'Speculated {first_mismatch_abs} token successfully')
                    start_pos = input_tensor.shape[1] + first_mismatch_abs
                    n_steps = input_with_mask_tensor.shape[1] - start_pos - 1
                    return self.autoregressive_decoding(input_with_mask_tensor[:, :(input_tensor.shape[1]+1+first_mismatch_abs)], n_steps)
                    
                else:
                    return input_with_mask_tensor
