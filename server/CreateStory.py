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
        "너는 지금부터 이야기 작가야.\n"
        "프롬프트를 기반으로 세계관(world), 등장인물(characters), 짧은 줄거리(plot), 이야기 진행(story_progression), 태그(tags)를 json형태로 출력해야해, 단, 캐릭터와 태그는 여려개여도 괜찮아. :\n"
        "프롬프트:" + {data} + "\n"
        "스토리:"
    )


def generate_story(data):
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

