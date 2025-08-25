from transformers import AutoTokenizer
import torch

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

def test_hypothesis(model, input_text, correct_answer):
    device = model.base_model.device

    # Get placeholder tokens - but we need enough placeholders to generate the answer
    correct_tokens = model.tokenizer(correct_answer, add_special_tokens=False)['input_ids']
    placeholder_tokens = model.tokenizer("-", add_special_tokens=False)['input_ids'] * len(correct_tokens)

    input_tokens = model.tokenizer(input_text)['input_ids']

    # Create hypothesis input: original input + enough placeholders for the answer
    hypothesis_input = input_tokens + placeholder_tokens
    hypothesis_input_tensor = torch.tensor(hypothesis_input).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model.base_model(
            input_ids=hypothesis_input_tensor,
            output_hidden_states=True,
            use_cache=False
        )

    logits = outputs.logits[0]  # Shape: (sequence_length, vocab_size)
    input_length = len(input_tokens)

    print(f"\nTesting hypothesis: '{input_text}' → '{correct_answer}'")
    print("=" * 100)

    total_tokens = len(correct_tokens)
    for i, correct_token_id in enumerate(correct_tokens):
        # Position in logits that should predict this token
        # We want to predict the i-th token of the answer at position input_length + i - 1
        pos = input_length + i - 1

        # Safety check
        if pos >= logits.shape[0]:
            print(f"  Token {i+1:2d}/{total_tokens}: ERROR - Position {pos} out of bounds (logits shape: {logits.shape})")
            continue

        position_logits = logits[pos]

        # Find ranking
        sorted_indices = torch.argsort(position_logits, descending=True)
        ranking = (sorted_indices == correct_token_id).nonzero(as_tuple=True)[0].item() + 1

        # Get token text and clean it up for display
        token_text = model.tokenizer.decode([correct_token_id])

        # Format ranking
        rank_display = f"#{ranking}"

        # Display token with proper escaping for special characters
        if token_text.strip() == "":
            if token_text == "\n":
                display_token = "\\n"
            elif token_text == "\t":
                display_token = "\\t"
            elif token_text == " ":
                display_token = "⎵"  # visible space character
            else:
                display_token = repr(token_text)
        else:
            display_token = f"'{token_text}'"

        print(f"  Token {i+1:2d}/{total_tokens}: {display_token:15s} → {rank_display:8s}")

    print("=" * 100)
