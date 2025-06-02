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
        "world": "에르델 마법의 숲",
        "genre": "판타지",
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
        "plot": "에르델 마법의 숲에서 사계절이 동시에 존재하는 정령들이 살고 있다. 어느 날, 숲에서 이상한 일이 일어나고, 정령들은 이를 해결하기 위해 모여든다. 하지만, 각 정령들은 서로 다른 능력과 성격으로 인해 의견이 분분하다. 결국, 정령들은 함께 협력하여 문제를 해결하고, 각자의 능력을 더욱 발전시켜 나간다.",
        "story_progression": "정령들은 각자의 능력을 발휘하여 문제를 해결하려 하지만, 서로 다른 의견으로 인해 갈등이 발생한다. 하지만, 정령들은 서로를 이해하고 존중하며, 함께 문제를 해결하는 방법을 찾아낸다. 이를 통해 정령들은 더욱 친밀해지고, 각자의 능력을 더욱 발전시켜 나간다.",
        "tags": ["자연", "협력", "성장"]
}

출력 : 
{
    "pages_text": [
        {
            "number": 1,
            "text": "1페이지 내용이 여기에 들어갑니다. 이 페이지는 이야기의 시작을 나타냅니다."
        },
        {
            "number": 2,
            "text": "2페이지 내용이 여기에 들어갑니다. 이 페이지는 이야기의 전개를 나타냅니다."
        },
        {
            "number": 3,
            "text": "3페이지 내용이 여기에 들어갑니다. 이 페이지는 이야기의 갈등을 나타냅니다."
        },
        {
            "number": 4,
            "text": "4페이지 내용이 여기에 들어갑니다. 이 페이지는 이야기의 절정을 나타냅니다."
        },
        {
            "number": 5,
            "text": "5페이지 내용이 여기에 들어갑니다. 이 페이지는 이야기의 결말을 나타냅니다."
        }
    ]
}

규칙 : 
- 1페이지부터 5페이지까지 전부 다 작성해야 함.
- output은 한글로 답해야 함
- 출력 구조대로 써야 함
- python 코드 등 코드형식이 아니라 반드시 JSON형식으로 출력해야 함

위 입력, 규칙, 출력을 참고해서 페이지당 구체적인 내용을 작성을 하도록 해줘
""" 

story = generate_story(prompt)
print(story[len(prompt):])
