from .train import build_trainer
from datasets import load_dataset
from transformers import TrainingArguments
import yaml

class QwenTrainer:
    def __init__(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def train(self):
        raw_dataset = load_dataset("json", data_files=self.config["train_data_dir"])
        dataset = raw_dataset["train"]  # ✅ 반드시 split 선택

        training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            per_device_train_batch_size=self.config["batch_size"],
            num_train_epochs=self.config["epochs"],
            logging_dir=self.config.get("logging_dir", "./logs"),
            save_steps=100,
        )

        trainer = build_trainer(self.config, dataset, training_args)
        trainer.train()