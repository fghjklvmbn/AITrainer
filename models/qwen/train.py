from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
import evaluate
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

# 장치 설정 (맥 용 MPS 혹은 Nvidia GPU 사용 가능 시 사용)
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
    
        # BLEU 계산
        bleu = evaluate.load("bleu")
        # BLEU는 references를 [list of list of tokens] 형식으로 요구
        # 예: references = [[ref1_tokens], [ref2_tokens], ...]
        references = [label.split() for label in decoded_labels]  # ✅ 수정: [[label.split()]] → [label.split()]
        predictions_tokenized = [pred.split() for pred in decoded_preds]
    
        # ROUGE 계산
        rouge = evaluate.load("rouge")
    
        # BLEU 계산
        bleu_result = bleu.compute(predictions=predictions_tokenized, references=references)
        # ROUGE 계산
        rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    
        # BLEU의 "bleu" 키가 존재하는지 확인 (예: "bleu1", "bleu2", "bleu3", "bleu4" 등)
        # ROUGE의 "rougeL" 키가 존재하는지 확인
        return {
            "bleu": bleu_result.get("bleu", 0.0),  # 기본값 설정
            "rougeL": rouge_result.get("rougeL", {}).get("mid", {}).get("fmeasure", 0.0)
        }

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
            max_length=config.get("max_length", 512)
        )
        tokenized = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=128)
        inputs["labels"] = tokenized["input_ids"]
        return inputs
    
    # train_dataset = 
    
    # tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_train_dataset = train_dataset["train"].map(tokenize_function, batched=False)
    tokenized_val_dataset = val_dataset["test"].map(tokenize_function, batched=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
    )
    
    return trainer, model, tokenizer
