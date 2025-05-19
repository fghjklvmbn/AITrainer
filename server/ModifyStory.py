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

# 동화의 줄거리를 바탕으로 문장을 재구성 하는 프롬프트
def format_prompt(data1, data2):
    return (
        "너는 지금부터 이야기 작가야.\n"
        "프롬프트를 기반으로 문장을 수정해서 json형태로 출력해야되 :\n"
        "전체 스토리 : " + {data1} + "\n"
        "프롬프트 : " + {data2} + "\n"
        "결과 :"
    )


def modify_story(data1, data2):
    prompt = format_prompt(data1, data2)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

