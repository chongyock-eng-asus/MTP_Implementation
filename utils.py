from transformers import AutoTokenizer

def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config['model_basename'], model_max_length=config['max_length'], token=config['API_KEY'])

    # Create new mask tokens
    mask_tokens = [f"<mask_{i}>" for i in range(config['num_masks'])]

    # Adding mask tokens into the tokenizer
    tokenizer.add_tokens(mask_tokens)
    
    mask_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in mask_tokens]
    tokenizer.custom_mask_tokens = mask_tokens
    tokenizer.custom_mask_token_ids = mask_token_ids
    return tokenizer


