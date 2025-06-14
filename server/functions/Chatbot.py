import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = None
model = None

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        model_path = "Qwen/Qwen3-1.7B"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
    return tokenizer, model

def strip_think_tags(text: str) -> str:
    return re.sub(r"user.*?</think>\n*", "", text, flags=re.DOTALL)

def chat(user_input):
    tokenizer, model = load_model()

    # 단일 메시지를 포맷에 맞춰 변환
    messages = [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        temperature=0.5,
        max_new_tokens=2048
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return strip_think_tags(response)