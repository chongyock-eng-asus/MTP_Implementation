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
        self.lora_A = nn.Parameter(torch.randn(rank, base_layer.in_features) * 0.01, device=device) # [rank, in_features]
        self.lora_B = nn.Parameter(torch.randn(base_layer.out_features, rank) * 0.01, device=device) # [out_features, rank]
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

        # Share embedding weights with sampler head output projection
        self.sampler_head.output_projection.weight = self.base_model.lm_head.weight

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
        position_ids: Optional[torch.Tensor]=None,
    ) -> Dict[str, torch.Tensor]:

        # Introduce mtp_mask hooks here
        self.current_mtp_mask = mtp_mask

        # Forward pass through base model
        outputs = self.base_model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False
        )
        logits = outputs.logits
        hidden_state = outputs.hidden_states[-1]

        base_loss = self._calculate_base_loss(logits, labels, mtp_mask)
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

    def _calculate_base_loss(self, logits: torch.Tensor, labels:torch.Tensor, mtp_mask: torch.Tensor):
        masked_labels = labels.clone()
        # masked_labels[~mtp_mask] = -100
        base_loss = F.cross_entropy(
        logits.view(-1, self.vocab_size), # [batch * seq_len, vocab_size]
        masked_labels.view(-1), # [seq_len]
        ignore_index=-100)
        return base_loss


    def _calculate_sampler_loss(self, input_ids:torch.Tensor, hidden_state: torch.Tensor, mtp_mask: torch.Tensor, labels:torch.Tensor):
        # Get embeddings for input tokens
        embedding = self.base_model.get_input_embeddings()
        shifted_input_ids = torch.cat([torch.zeros_like(input_ids[:, :1]), input_ids[:, :-1]], dim=1)
        prev_embeddings = embedding(shifted_input_ids)

        mtp_positions = mtp_mask.bool()
        mtp_hidden = hidden_state[mtp_positions]
        mtp_prev_emb = prev_embeddings[mtp_positions]
        mtp_sampler_logits = self.sampler_head(mtp_hidden, mtp_prev_emb) # [num_mtp_tokens, vocab_size]
        mtp_labels = labels[mtp_positions] # [num_mtp_tokens]

        sampler_loss =  F.cross_entropy(
                mtp_sampler_logits, # [num_mtp_tokens, vocab_size]
                mtp_labels,  # [num_mtp_tokens]
                ignore_index=-100)

        return sampler_loss

    def _calculate_lcm_loss(self, position_mask:torch.Tensor, hidden_state:torch.Tensor, mtp_mask:torch.Tensor):
        true_positions = torch.where(~mtp_mask)[1]
        total_lcm_loss = torch.tensor(0.0, device=self.base_model.device)
        for i in true_positions:
            if i !=0:
                current_ntp_pos = i.item()
                current_pos_id = position_mask[0][i]
                # For every anchor, find index with the same positon id
                target_idx = (position_mask[0, :current_ntp_pos] == current_pos_id).nonzero(as_tuple=True)[0]
                length_set = target_idx.shape[0]
                if length_set > 0:
                    lcm_loss = 0
                    ntp_representation = hidden_state[0][i]

                    for item in target_idx:
                        prev_representation = hidden_state[0][item]
                        lcm_loss += torch.mean((ntp_representation - prev_representation) ** 2)
                    total_lcm_loss += lcm_loss/length_set

        return total_lcm_loss/len(true_positions)

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
        print(f"Input tensor:   {input_tensor}")
        masked_token_ids = self.tokenizer.custom_mask_token_ids
        masked_token_tensor = torch.tensor(masked_token_ids).unsqueeze(0).to(device)
        current_mtp_idx = input_tensor.shape[1]

        input_with_mask_tensor = torch.concat([input_tensor, masked_token_tensor],dim=-1)
        print(f"Input tensor with mask: {input_with_mask_tensor}")

        mtp_mask = torch.concat([torch.zeros(input_tensor.shape[0], input_tensor.shape[1], device=device),
                        torch.ones(masked_token_tensor.shape[0], masked_token_tensor.shape[1], device=device)], dim=-1).bool()
        print(f"MTP mask: {mtp_mask}")

        self.current_mtp_mask = mtp_mask # [batch, seq_len]
        with torch.no_grad():
            outputs = self.base_model(
                    input_ids=input_with_mask_tensor,
                    output_hidden_states=True,
                    use_cache=False
                )
            hidden_states = outputs.hidden_states[-1]
            next_token = torch.argmax(outputs.logits, dim=-1)[:, current_mtp_idx-1].unsqueeze(0)
            print(f"Predicted next token: {next_token}")

            input_with_mask_tensor[:,current_mtp_idx] = next_token
            print(f"Input tensor after first prediction: {input_with_mask_tensor}")
            # Sample the remaining tokens
            shifted_input_ids = torch.cat([torch.zeros_like(input_with_mask_tensor[:, :1]), input_with_mask_tensor[:, :-1]], dim=1)

            embedding = self.base_model.get_input_embeddings()
            prev_embeddings = embedding(shifted_input_ids)
            mtp_mask[:,current_mtp_idx] = 0 # since next predicted token is verified
            mtp_positions = mtp_mask.bool()
            mtp_hidden = hidden_states[mtp_positions]
            mtp_prev_emb = prev_embeddings[mtp_positions]
            mtp_sampler_logits = self.sampler_head(mtp_hidden, mtp_prev_emb)
            next_tokens = torch.argmax(mtp_sampler_logits, dim=-1).unsqueeze(0) # [batch, seq_len]
            print(f"Input tensor after sampling: {next_tokens}")

            input_with_mask_tensor[:,-(self.num_masks-1):] = next_tokens
            print(f"Predicted full tokens: {input_with_mask_tensor}")


        self.current_mtp_mask = None

        # Verification process
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
