from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json

# 1. 사용할 모델명 설정 (예: 3B 모델)
MODEL_NAME = "Qwen/Qwen2.5-3B"  # 또는 임의의 3B 모델로 변경

# 2. 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# 3. 생성 파이프라인 정의
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 4. 테스트용 반복 프롬프트 정의
base_prompt = """다음 내용을 기반으로 어린이용 동화 설정 정보를 JSON 형식으로 생성해줘.

프롬프트 :  
"사계절이 동시에 존재하는 마법의 숲, 에르델에는 봄, 여름, 가을, 겨울의 정령이 살고 있어, 여기서 무슨일이 일어나는데, 그것에 대한 과정의 동화"

구조 : 
{
  "world": "세계관 설명",
  "genre": "동화, 어린이, 판타지 등",
  "characters": [
    {
      "character_name": "이름",
      "gender": "성별",
      "personality": "성격",
      "ability": "능력",
      "main_character": true 또는 false
    },
    ...
  ],
  "plot": "3~5줄 줄거리",
  "story_progression": "3~5줄 이야기 전개",
  "tags": ["태그1", "태그2", "태그3"]
}

규칙:
- 세계관은 위 프롬프트 내용을 기반으로 작성
- 장르는 "판타지", "우화", "교훈", "모험", "생태", "감성", "교육" 중에 관련있는 것을 하나만 선택
- 등장인물은 어린이 이해 가능한 성격과 능력 포함
- 등장인물은 2~5명, 성격은 3~5개 단어로 작성
- plot은 3~5줄, story_progression도 3~5줄
- tags는 반드시 3개
- 능력은 상황에 따라 다르게 활용되어야 하며, 캐릭터는 유머러스하고 매력 있어야 함
- 출력은 반드시 JSON 형식으로 출력
"""

# 5. 반복 생성 실행
NUM_SAMPLES = 10
responses = []

for i in range(NUM_SAMPLES):
    print(f"\n=== Sample {i+1} ===")
    output = generator(
        base_prompt,
        max_new_tokens=1024,
        do_sample=True,
        top_k=50,
        temperature=0.8,
        num_return_sequences=1
    )
    text = output[0]["generated_text"].replace(base_prompt, "").strip()
    print(text)
    responses.append(text)
    


def extract_valid_json(text: str):
    """
    주어진 텍스트에서 JSON 형식으로 보이는 부분만 추출하고,
    구조 조건을 만족하는 경우에만 리턴한다.
    """
    try:
        # 첫 번째 중괄호부터 추출
        json_start = text.find("{")
        if json_start == -1:
            return None

        json_candidate = text[json_start:]
        
        # 중괄호 짝 맞추기: 유효 JSON 범위 추정
        brace_stack = []
        end_index = -1
        for i, char in enumerate(json_candidate):
            if char == '{':
                brace_stack.append('{')
            elif char == '}':
                if brace_stack:
                    brace_stack.pop()
                if not brace_stack:
                    end_index = i + 1
                    break
        if end_index == -1:
            return None

        json_str = json_candidate[:end_index]

        # JSON 파싱
        parsed = json.loads(json_str)

        # 필수 필드 확인
        required_fields = ["world", "genre", "characters", "plot", "story_progression", "tags"]
        if not all(field in parsed for field in required_fields):
            return None

        # characters 검사
        if not isinstance(parsed["characters"], list) or len(parsed["characters"]) == 0:
            return None

        for character in parsed["characters"]:
            if not all(k in character for k in ["character_name", "gender", "personality", "ability", "main_character"]):
                return None
            if not isinstance(character["main_character"], bool):
                return None

        # tags 검사
        if not isinstance(parsed["tags"], list) or len(parsed["tags"]) != 3:
            return None

        return parsed

    except Exception:
        return None