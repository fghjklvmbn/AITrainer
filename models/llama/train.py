from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
import evaluate


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def build_trainer(config, train_dataset, val_dataset, training_args):
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
        use_fast=False,
    )

    # --- 평가 지표 계산 함수
    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     loss_fn = torch.nn.CrossEntropyLoss()
    #     logits = torch.tensor(logits).to(DEVICE)
    #     labels = torch.tensor(labels).to(DEVICE)
    #     loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    #     return {"eval_loss": loss.item()}

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # 디코딩
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 텍스트 후처리 (공백 제거 등)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # BLEU/ROUGE 계산
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")

        bleu_result = bleu.compute(predictions=[pred.split() for pred in decoded_preds],
                                   references=[[label.split()] for label in decoded_labels])
        rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            "bleu": bleu_result["bleu"],
            "rougeL": rouge_result["rougeL"].mid.fmeasure
        }



    # --- LoRA 설정
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.get("lora_r", config["lora_r"]),
        lora_alpha=config.get("lora_alpha", config["lora_alpha"]),
        target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"]),
        lora_dropout=config.get("lora_dropout", config["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 트레이닝 가능한 파라미터(LoRA) 출력
    model.print_trainable_parameters()

    # --- Tokenization
    def tokenize_function(examples):
        # prompt 형식 구성
        prompt = examples["instruction"] + " " + examples["input"]
        inputs = tokenizer(
            prompt, 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
        # 토크나이즈
        tokenized = tokenizer(
            examples["output"],
            truncation=True,
            padding="max_length",
            max_length=config.get("max_length", 128)
        )
        tokenized = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=128)
        inputs["labels"] = tokenized["input_ids"]
        return inputs

    tokenized_train_dataset = train_dataset["train"].map(tokenize_function, batched=False)
    tokenized_val_dataset = val_dataset["test"].map(tokenize_function, batched=False)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
    )

    
    return trainer, tokenizer, model
