from .train import build_trainer
from datasets import load_dataset
from transformers import TrainingArguments
import yaml

class QwenTrainer:
    def __init__(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def train(self):
        dataset = load_dataset("json", data_files = self.config["train_data_dir"])

        training_args = TrainingArguments(
            # 기본설정
            output_dir=self.config["output_dir"],
            per_device_train_batch_size=self.config["batch_size"],
            num_train_epochs=self.config["epochs"],
            learning_rate=self.config["learning_rate"],
            fp16=self.config.get("fp16", False),  # 맥 혹은 cpu용으로 돌릴려면 False 혹은 주석 필요
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 1),
            warmup_ratio=self.config.get("warmup_ratio", 0.1),
            
            # datealoader 관련 설정
            dataloader_num_workers=self.config.get("dataloader_num_workers", 4),
            dataloader_pin_memory=self.config.get("dataloader_pin_memory", True),
            dataloader_drop_last=self.config.get("dataloader_drop_last", True),


            # 옵티마이저 관련 설정
            optim=self.config.get("optim", "adamw_torch"),
            lr_scheduler_type=self.config.get("lr_scheduler_type", "linear"),
            warmup_steps=self.config.get("warmup_steps", 0),
            max_steps=self.config.get("max_steps", -1),
            # optimizers=(self.config.get("optimizer"), self.config.get("lr_scheduler")),

            # 저장/로깅 관련 설정
            save_steps=self.config.get("save_steps", 500),
            save_total_limit=self.config.get("save_total_limit", 2),
            evaluation_strategy="steps",
            eval_steps=self.config.get("eval_steps", 500),
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_steps_per_second",
            greater_is_better=True,

            # 로깅 관련 설정
            logging_dir=self.config.get("logging_dir", "./logs"),
            report_to="tensorboard",
            logging_steps=self.config.get("logging_steps", 10),
            logging_first_step=self.config.get("logging_first_step", True),


            # 기타 설정
            remove_unused_columns=self.config.get("remove_unused_columns", False),
            label_names=self.config.get("label_names", ["labels"]),
            save_on_each_node=self.config.get("save_on_each_node", True),
        )

        trainer = build_trainer(self.config, dataset, training_args)
        trainer.train()