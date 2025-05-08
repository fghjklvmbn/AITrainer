# models/qwen/train.py

from utils.data_loader import get_dataloader
from transformers import AutoTokenizer

def prepare_training(config):
    # 예시: 데이터 로딩 및 전처리
    print("[QWEN] Preparing data...")

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    train_loader = get_dataloader(
        data_path=config["train_data"],
        tokenizer=tokenizer,
        batch_size=config["batch_size"]
    )

    return train_loader, tokenizer
