from models.llama.trainer import LlamaTrainer
from models.qwen.trainer import QwenTrainer

def get_trainer(model_type: str, config_path: str):
    """
    model_type에 따라 적절한 Trainer 인스턴스를 반환.
    """
    if model_type.lower() == "llama":
        return LlamaTrainer(config_path)
    elif model_type.lower() == "qwen":
        return QwenTrainer(config_path)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
