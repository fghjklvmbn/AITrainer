import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

# 기본 모델(Base Model) 경로
base_model_path = "Qwen/Qwen2.5-3B"  # 기본 모델 경로 (예: Llama 7B)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 기본 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="cuda"
)

# Adapter Model 로드
# model = PeftModel.from_pretrained(model, "./storybook_model")

# # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device="cuda"
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
사계절이 동시에 존재하는 마법 숲 '에르델'을 배경으로, 봄·여름·가을·겨울의 정령들이 나오는 짧은 동화를 만들어줘.  
- 어린이용 이야기야.  
- 등장인물의 이름, 성격, 능력은 간단히 소개해줘.  
- 줄거리와 이야기 전개는 3~5줄로 써줘.  
- 전체 이야기를 JSON 형식으로 만들어줘.
""" 

# 다음 내용을 기반으로 어린이용 동화 설정 정보를 JSON 형식으로 생성해줘
# 프롬프트 : 
# "사계절이 동시에 존재하는 마법의 숲, 에르델에는 봄, 여름, 가을, 겨울의 정령이 살고 있어, 여기서 무슨일이 일어나는데, 그것에 대한 과정의 동화"

# 구조 :
# {
#   "world": "세계관 설명",
#   "genre": "동화, 어린이, 판타지 등",
#   "characters": [
#     {
#       "character_name": "이름",
#       "gender": "성별",
#       "personality": "성격",
#       "ability": "능력",
#       "main_character": true 또는 false
#     },
#     ...
#   ],
#   "plot": "3~5줄 줄거리",
#   "story_progression": "3~5줄 이야기 전개",
#   "tags": ["태그1", "태그2", "태그3"]
# }

# 규칙:
# - 세계관은 위 프롬프트 내용을 기반으로 작성
# - 장르는 반드시 '동화', '어린이' 포함
# - 등장인물은 어린이 이해 가능한 성격과 능력 포함
# - plot은 3~5줄, story_progression도 3~5줄
# - tags는 반드시 3개
# - 능력은 상황에 따라 다르게 활용되어야 하며, 캐릭터는 유머러스하고 매력 있어야 함
# - 출력은 반드시 JSON 형식으로 출력
# 프롬프트 : "사계절이 동시에 존재하는 마법의 숲, 에르델에는 봄, 여름, 가을, 겨울의 정령이 살고 있어, 여기서 무슨일이 일어나는데, 그것에 대한 과정의 동화" 를 

# 구조 : 
# {
# 	"world": "세계관"
#     "genre": "장르",
# 	"characters" : [
# 		{ 
#   		"character_name": "등장인물1 이름", "등장인물2 이름" ...
# 		"gender": "등장인물1 성별", "등장인물2 성별" ...  
# 		"personality": "등장인물1 성격", "등장인물2 성격" ...
# 		"ability": "등장인물1의 능력", "등장인물2의 능력" ...
# 		"main_character": "등장인물의 주인공 여부"
#   		}, ...
# 	]
# 	"plot": "대략적인 줄거리"
# 	"story_progression": "대략적인 스토리 전개방향"
# 	"tags": [태그1, 태그2, 태그3, ...]
# }

# 규칙 : 
# - 이전에 작성된 내용의 요약을 바탕으로 세계관을 생성
# - 장르는 동화, 어린이, 어린이가 이해할 수 있는 이야기
# - 등장인물들은 어린이들이 이해할 수 있는 성격, 능력, 이름, 등장 배경 등을 포함
# - plot(줄거리)는 3~5줄로 작성
# - story_progression(이야기 전개 과정)는 3~5줄로 작성
# - tags(상징하는 태그, 3개)는 3개로 작성
# - 동화의 특징은 전개가 자연스럽고, 이야기가 흥미롭고, 캐릭터가 유머러스하고, 이야기가 흥미로운, 상징적인 요소가 포함되어야 한다
# - 캐릭터의 능력은 상황에 따라 다르게 활용되어야 한다
# - 등장인물의 능력은 상황에 따라 다르게 활용되어야 한다

# 위 구조와 규칙을 토대로 프롬프트의 글을 토대로 
# json형식으로 작성해줘

story = generate_story(prompt)
print(story[len(prompt):])