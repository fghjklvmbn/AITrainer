import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, AutoConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from data.data_all import data
import argparse
import utils

# 장치 설정 (맥 용 MPS 혹은 Nvidia GPU 사용 가능 시 사용)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# yaml 파징
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)

# parse commands
cmd = parse_args()
# load config
args = utils.load_config(cmd.config)

# Checkpoint 찾기 함수
def get_last_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    # 최신 checkpoint 리턴
    return max(checkpoints, key=lambda x: int(x.split('-')[-1]))

def numcheck(checkpoint_name):
    checkpoint = os.path.basename(checkpoint_name)
    if not checkpoint.startswith("checkpoint-"):
        # 형식이 맞지 않으면 None 반환
        return 0
    try:
        # "checkpoint-" 이후의 숫자 부분 추출
        number = int(checkpoint.split('-')[-1])
        return number
    except ValueError:
        # 숫자로 변환할 수 없는 경우 None 반환
        return 0

def inputepoch(num, last_checked):
    print(f"학습될 epoch 기본값 : {num}");
    print(f"현재 학습된 checkpoint 값 : {last_checked}");
    input_num = input("더 학습할 epoch 값을 입력해주세요: ")
    # 엔터시
    if input_num == '':
        print("epoch 기본값으로 실행합니다.")
        return num + last_checked
    # 문자일 경우
    elif not input_num.isdigit():
        print("잘못된 값입니다. 설정한 epoch(250) 만큼 학습한 다음 종료 합니다. 나중에 다시 실행해주세요")
        return last_checked
    # 숫자일 경우
    else :
        sum = int(input_num) + last_checked
        # 혹시모를 현변환
        sum = int(sum)
        print(f"{int(input_num)} epoch 값 만큼 추가 학습합니다.")
        return sum

# OUTPUT_DIR가 없으면 생성
os.makedirs(args.train.output_dir, exist_ok=True)

# 마지막 체크포인트 확인
last_checkpoint = get_last_checkpoint(args.train.output_dir)

if last_checkpoint == None:
    print("✅ 체크포인트가 없습니다. 새로 학습을 시작합니다.")
    num_of_epoch=5
else:
    print(f"✅ 체크포인트 발견: {last_checkpoint}. 이어서 학습합니다.")


# 모델 로드
if last_checkpoint:
    print("🔄 체크포인트에서 모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(last_checkpoint).to(DEVICE)
    last_epoch = numcheck(last_checkpoint)
    num_of_epoch = inputepoch(250, last_epoch)
else:
    print("🌱 사전 훈련된 모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(args.train.MODEL_NAME).to(DEVICE)

# LoRA 설정 및 모델 적용
lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(args.train.MODEL_NAME)

# 데이터셋 구성
dataset = Dataset.from_list(data)
split_data = dataset.train_test_split(test_size=0.0003)
print(dataset)

def tokenize_function(example):
    prompt = example["instruction"] + " " + example["input"]
    inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=128)
    labels = tokenizer(example["output"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_train_dataset = split_data["train"].map(tokenize_function)
tokenized_val_dataset = split_data["test"].map(tokenize_function)

# 평가 지표 계산 함수
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    loss_fn = torch.nn.CrossEntropyLoss()
    logits = torch.tensor(logits).to(DEVICE)
    labels = torch.tensor(labels).to(DEVICE)
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    return {"eval_loss": loss.item()}

# 학습 설정
training_args = TrainingArguments(
    output_dir=args.train.OUTPUT_DIR,
    per_device_train_batch_size=args.train.batch_size,
    num_train_epochs=int(num_of_epoch),
    logging_steps=args.train.logging_steps,
    save_steps=args.train.save_steps,
    save_total_limit=args.train.save_total_limit,
    learning_rate=args.train.lr,
    fp16=args.train.fp16,  # 맥 혹은 cpu용으로 돌릴려면 False 혹은 주석 필요
    gradient_accumulation_steps=args.train.gradient_accumulation_steps,
    metric_for_best_model="eval_steps_per_second",
    warmup_ratio=args.train.warmup_ratio,
    eval_steps=args.train.eval_steps,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
)

# Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_metrics,
)

# 학습 시작 (체크포인트에서 이어서 학습)
trainer.train(resume_from_checkpoint=last_checkpoint)

# 모델 저장
model.save_pretrained(args.train.LOCAL_MODEL_PATH)
tokenizer.save_pretrained(args.train.LOCAL_MODEL_PATH)

print("🎉 모델이 성공적으로 저장되었습니다!")