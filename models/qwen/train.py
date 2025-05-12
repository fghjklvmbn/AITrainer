from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

def build_trainer(config, dataset, training_args):
    # --- 모델 및 토크나이저 로딩
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=True,
    )

    # --- LoRA 설정
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"]),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 평가 지표 계산 함수
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        loss_fn = torch.nn.CrossEntropyLoss()
        logits = torch.tensor(logits).to(DEVICE)
        labels = torch.tensor(labels).to(DEVICE)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"eval_loss": loss.item()}

    # --- Tokenization
    # TODO: 베이스를 기반으로 tokenizer를 설정할 수 있도록 수정필요 
    def tokenize_function(examples):
        # prompt 형식 구성
        dataset = dataset.from_list(data)
        split_data = dataset.train_test_split(test_size=0.0003)
        prompts = []
        for instr, inp in zip(examples["instruction"], examples["input"]):
            if inp:
                prompt = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instr}\n\n### Response:\n"
            prompts.append(prompt)
    
        # 정답과 연결 (Prompt + Output)
        full_texts = [p + o for p, o in zip(prompts, examples["output"])]

        # 토크나이즈
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            padding="max_length",
            max_length=config.get("max_length", 512)
        )

        # labels로 input_ids 복사
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_train_dataset = split_data["train"].map(tokenize_function)
    tokenized_val_dataset = split_data["test"].map(tokenize_function)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer
