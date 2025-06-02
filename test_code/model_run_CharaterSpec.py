import os
os.environ["HF_HOME"] = "/data/wonderland/beta/huggingface/"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

# 기본 모델(Base Model) 경로
base_model_path = "Qwen/Qwen2.5-32B"  # 기본 모델 경로 (예: Llama 7B)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 기본 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="cpu"
)

# Adapter Model 로드
# model = PeftModel.from_pretrained(model, "./storybook_model")

# # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device="cpu"
# model.to(device) 

def generate_story(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 테스트
prompt = """
입력 : 
{
		"page" : 1,
		"detail": "에르델 마법의 숲에서 사계절이 동시에 존재하는 정령들이 살고 있습니다. 봄 정령, 여름 정령, 가을 정령, 겨울 정령이 모두 함께 살고 있습니다. 어느 날, 숲에서 이상한 일이 일어나고, 정령들은 이를 해결하기 위해 모여듭니다."
}

출력구조 :  
{
  "image_prompt" :
    {
      "page": 1,
      "prompt": ""
    }
}

규칙 : 
- 캐릭터는 저연령층 타겟으로 있도록 해야함
- prompt는 영어로 작성되어야만 함
- 페이지에 맞는 묘사를 해서 stable diffusion 같은 이미지 생성 모델에 처리하기 최적화 하도록 작성해야함
- python 코드 등 코드형식이 아니라 반드시 JSON형식으로 출력해야 함

위 입력 그리고 규칙을 참고해서 출력으로 구체적인 묘사를 하는 출력구조로 만들어줘
""" 

story = generate_story(prompt)
print(story[len(prompt):])
