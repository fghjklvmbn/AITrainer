import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

# 기본 모델(Base Model) 경로
base_model_path = "Qwen/Qwen3-1.7B"  # 기본 모델 경로 (예: Llama 7B)

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
        max_length=1024,
        temperature=0.5,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 테스트
prompt = """
푸른 숲의 끝자락에 위치한 ‘엘로라’라는 마을은 자연의 정령들과 사람이 조화를 이루며 사는 곳입니다. 이 마을에 사는 리아는 식물과 대화를 나눌 수 있는 특별한 소녀인데, 최근 숲속에서 이상한 어둠이 퍼지며 나무들이 죽어가고 있어요. 리아는 친구 루크(용감한 소년, 동물과 마음을 나눌 수 있음), 미로(장난기 많은 바람 마법사)와 함께 숲을 구하기 위한 모험을 시작합니다." \
를 
world(세계관)
genre(장르, 1가지)
characters(등장인물들)
(
character_name(등장인물 이름)
main_character(주인공 여부)
gender(성별)
personality(성격)
ability(능력)
)
plot(줄거리)
story_progression(이야기 전개 과정)
tags(상징하는 태그, 3개)

이 구조로 json을 바탕으로 5개 정도
한국어로 작성해줘
""" 
story = generate_story(prompt)
print(story)