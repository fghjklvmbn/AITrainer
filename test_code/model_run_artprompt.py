from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# 1. 사용할 모델명 설정
MODEL_NAME = "Qwen/Qwen2.5-32B"

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
base_prompt = """
입력 : 
{
	"detail": "에르델 마법의 숲에서 사계절이 동시에 존재하는 정령들이 살고 있습니다. 봄 정령, 여름 정령, 가을 정령, 겨울 정령이 모두 함께 살고 있습니다. 어느 날, 숲에서 이상한 일이 일어나고, 정령들은 이를 해결하기 위해 모여듭니다."
}

출력구조 :
{
	"image_prompt" : 영문 프롬프트
}

규칙 :
- 저연령층이 볼수 있도록 작성해야함
- prompt는 영어로 작성되어야만 함
- 페이지에 맞는 묘사를 해서 stable diffusion 같은 이미지 생성 모델에 처리하기 최적화 하도록 작성해야함
- python 코드 등 코드형식이 아니라 반드시 JSON형식으로 출력해야 함

위 입력, 출력구조 그리고 규칙을 참고해서 출력으로 구체적인 묘사를 하는 출력구조로 만들어줘
"""


# 6. 반복 실행
NUM_SAMPLES = 1
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

    
