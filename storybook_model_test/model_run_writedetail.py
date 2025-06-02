import os
os.environ["HF_HOME"] = "/data/wonderland/beta/huggingface/"

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# 1. 사용할 모델명 설정 (예: 3B 모델)
MODEL_NAME = "Qwen/Qwen2.5-32B"  # 또는 임의의 3B 모델로 변경

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
base_prompt = """
입력 :
{
	"createpage" : "5",
	"story_progression" : "정령들은 각자의 능력을 발휘하여 문제를 해결하려 하지만, 서로 다른 의견으로 인해 갈등이 발생한다. 하지만, 정령들은 서로를 이해하고 존중하며, 함께 문제를 해결하는 방법을 찾아낸다. 이를 통해 정령들은 더욱 친밀해지고, 각자의 능력을 더욱 발전시켜 나간다.",
    "plot": "에르델 마법의 숲에서 사계절이 동시에 존재하는 정령들이 살고 있다. 어느 날, 숲에서 이상한 일이 일어나고, 정령들은 이를 해결하기 위해 모여든다. 하지만, 각 정령들은 서로 다른 능력과 성격으로 인해 의견이 분분하다. 결국, 정령들은 함께 협력하여 문제를 해결하고, 각자의 능력을 더욱 발전시켜 나간다."
}

출력구조 :
{
    "pages_text": [
        {
            "number": 1,
            "text": 1페이지에 들어갈 내용
        },
        {
            "number": 2,
            "text": 2페이지에 들어갈 내용
        },
        {
            "number": 3,
            "text": 3페이지에 들어갈 내용
        },
        {
            "number": 4,
            "text": 4페이지에 들어갈 내용
        },
        {
            "number": 5,
            "text": 5페이지에 들어갈 내용
        }
    ]
}


규칙 :
- 문장을 저연령층 타겟으로 ~에요 나 ~했어요 등으로 문체를 적용하여 작성해야 함
- plot과 story_progression에 맞춰 스토리를 분할해야 함
- 이야기는 도입-문제제기-모험/여정-절정-해결-결말 순으로 5페이지에 다 맞도록 해야함
- 페이지 간 내용의 흐름에 맞춰야 함. 
- python 코드 등 코드형식이 아니라 반드시 JSON형식으로 출력해야 함

위 입력 그리고 규칙을 참고해서 출력으로 구체적인 묘사를 하는 출력구조로 만들어줘
"""

# 5. 반복 생성 실행
NUM_SAMPLES = 2 
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
    