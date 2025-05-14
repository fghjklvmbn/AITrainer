from trainer.trainer_beta import build_trainer
from utils.data_loader import load_dataset
from utils.utils_beta import set_random_seed
from config_loader import load_config
from transformers import TrainingArguments

def run_training():
    config = load_config('config/train_beta_qwen.yaml')

    set_random_seed(config.get('seed', 42))

    dataset = load_dataset('../data/json_files')

    training_args = TrainingArguments(
        output_dir="./checkpoint",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        logging_steps=100,
        learning_rate=2e-5,
        report_to="none",
    )

    trainer = build_trainer(config, dataset, training_args)
    trainer.train()