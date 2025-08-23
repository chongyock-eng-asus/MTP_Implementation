def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "model_basename": "meta-llama/Llama-3.2-1B",
        "num_masks": 3,
        "max_length": 20,
        "API_KEY":"",
        "lora_rank": 128, 
        "lora_alpha":256,
        "dataset_size": 5000,
        "datasource":"allenai/tulu-3-sft-mixture",
    }