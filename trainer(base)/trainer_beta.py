from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

def build_trainer(config, dataset, training_args):
    model = AutoModelForCausalLM.from_pretrained(
        config.get('model_name_or_path'),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.get('model_name_or_path'),
        trust_remote_code=True,
    )

    # --- LoRA 적용
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
        target_modules=config.get('lora_target_modules', ["q_proj", "v_proj"]),
        lora_dropout=config.get('lora_dropout', 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Tokenization
    def tokenize_function(examples):
        output = tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )
        output["labels"] = output["input_ids"].copy()
        return output

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # --- Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
    )

    trainer.train()