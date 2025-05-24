import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 기본 모델(Base Model) 경로
base_model_path = "Qwen/Qwen2.5-7B"  # 기본 모델 경로 (예: Llama 7B)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 기본 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Adapter Model 로드
model = PeftModel.from_pretrained(model, "./storybook_model")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

def format_prompt(data):
    return (
        "너는 프롬프트에 맞춰서 생성된 동화 등장인물의 상세정보를 JSON 형태로 출력해야해\n"
        "프롬프트에 해당하는 등장인물의 상세한 정보를 이름(name), 성별(gender), 취미(personality), 능력(ability), 생김새(appearance)를 출력해야돼 :\n"
        "프롬프트:" + {data} + "\n"
        "스토리:"
    )


def character_spec(data):
    prompt = format_prompt(data)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

