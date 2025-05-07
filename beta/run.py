# run_beta.py

import argparse
from models.qwen.trainer import QwenTrainer
from models.llama.trainer import LlamaTrainer

def get_trainer_class(model_type):
    if model_type == "qwen":
        return QwenTrainer
    elif model_type == "llama":
        return LlamaTrainer
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        import yaml
        config = yaml.safe_load(f)

    model_type = config.get("model_type")
    TrainerClass = get_trainer_class(model_type)
    trainer = TrainerClass(args.config)
    trainer.train()
