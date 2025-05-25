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
    "characters": [
        {
            "character_name": "봄 정령",
            "gender": "여성",
            "personality": "활기차고 사랑스럽다",
            "ability": "꽃을 피우고 새싹을 자라게 한다",
            "main_character": True
        },
        {
            "character_name": "여름 정령",
            "gender": "남성",
            "personality": "활발하고 에너지 넘치는",
            "ability": "태양의 힘을 빌려 열매를 열게 한다",
            "main_character": True
        },
        {
            "character_name": "가을 정령",
            "gender": "여성",
            "personality": "차분하고 따뜻한",
            "ability": "잎을 떨어뜨리고 수확을 돕는다",
            "main_character": True
        },
        {
            "character_name": "겨울 정령",
            "gender": "남성",
            "personality": "차갑고 진지한",
            "ability": "눈을 내리고 동물을 보호한다",
            "main_character": True
        }
    ],
}

출력구조 :  
{
  "image_prompts": [
    {
      "character_name": 캐릭터 이름,
      "prompt": 캐릭터에 맞게 묘사하는 문단 혹은 문장
    },
  ]
}

규칙 : 
- 캐릭터는 저연령층 타겟으로 있도록 해야함
- prompt는
- python 코드 등 코드형식이 아니라 반드시 JSON형식으로 출력해야 함

위 입력 그리고 규칙을 참고해서 출력으로 구체적인 묘사를 하는 출력구조로 만들어줘
""" 

story = generate_story(prompt)
print(story[len(prompt):])
