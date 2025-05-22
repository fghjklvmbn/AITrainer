from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json

# 1. 사용할 모델명 설정
MODEL_NAME = "Qwen/Qwen2.5-3B"

# 2. 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# 3. 생성 파이프라인 정의
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 4. 테스트용 프롬프트
base_prompt = """다음 내용을 기반으로 어린이용 동화 설정 정보를 JSON 형식으로 생성해줘.

프롬프트 :  
"사계절이 동시에 존재하는 마법의 숲, 에르델에는 봄, 여름, 가을, 겨울의 정령이 살고 있어, 여기서 무슨일이 일어나는데, 그것에 대한 과정의 동화"

구조 : 
{
  "world": "",
  "genre": "",
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
  "plot": "",
  "story_progression": "",
  "tags": ["", "", ""]
}

규칙:
- 세계관은 위 프롬프트 내용을 기반으로 작성
- 장르는 반드시 하나 만 포함
- 등장인물은 어린이 이해 가능한 성격과 능력 포함
- plot은 3~5줄, story_progression도 3~5줄
- tags는 반드시 3개
- 능력은 상황에 따라 다르게 활용되어야 하며, 캐릭터는 유머러스하고 매력 있어야 함
- 출력은 반드시 JSON 형식으로 출력
- 모든 키와 값 사이에는 쉼표가 있어야 함 (JSON 문법 오류 없이)
- 예시를 보여줄 때 JSON 구문 오류가 없도록 주의할 것
"""

# 5. JSON 추출 함수
def extract_valid_json(text: str):
    try:
        json_start = text.find("{")
        if json_start == -1:
            return None, "중괄호 시작 없음"

        json_candidate = text[json_start:]
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
            return None, "중괄호 짝이 맞지 않음"

        json_str = json_candidate[:end_index]
        parsed = json.loads(json_str)

        required_fields = ["world", "genre", "characters", "plot", "story_progression", "tags"]
        if not all(field in parsed for field in required_fields):
            return None, "필수 필드 누락"

        if not isinstance(parsed["characters"], list) or len(parsed["characters"]) == 0:
            return None, "characters 항목이 올바르지 않음"

        for character in parsed["characters"]:
            if not all(k in character for k in ["character_name", "gender", "personality", "ability", "main_character"]):
                return None, "character 필드 불완전"
            if not isinstance(character["main_character"], bool):
                return None, "main_character 값이 bool 아님"

        if not isinstance(parsed["tags"], list) or len(parsed["tags"]) != 3:
            return None, "tags 길이가 3이 아님"

        return parsed, "성공"
    except Exception as e:
        return None, f"예외 발생: {str(e)}"

# 6. 반복 실행
NUM_SAMPLES = 10
results = []

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

    response_text = output[0]["generated_text"]
    generated_part = response_text.replace(base_prompt, "").strip()

    parsed_result, status = extract_valid_json(response_text)

    if parsed_result:
        print("[✔] 유효 JSON 생성 성공")
        print(json.dumps(parsed_result, indent=2, ensure_ascii=False))
    else:
        print(f"[✘] 실패 이유: {status}")
        print("원본 출력 일부:", generated_part[:500], "..." if len(generated_part) > 500 else "")

    results.append({"index": i + 1, "success": parsed_result is not None, "reason": status})

# 7. 최종 요약
print("\n\n=== 최종 결과 요약 ===")
for r in results:
    status_str = "✔ 성공" if r["success"] else f"✘ 실패 - {r['reason']}"
    print(f"[{r['index']}] {status_str}")
