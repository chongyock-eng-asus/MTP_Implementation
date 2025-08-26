def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 1,
        "lr": 10**-4,
        "model_basename": "meta-llama/Llama-3.2-1B",
        "num_masks": 8,
        "max_length": 1024,
        "API_KEY":"",
        "lora_rank": 128, 
        "lora_alpha":256,
        "dataset_size": '',
        "datasource":"allenai/tulu-3-sft-mixture",
    }
