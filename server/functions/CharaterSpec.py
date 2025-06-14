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
    device_map="auto"
)

# Adapter Model 로드
# model = PeftModel.from_pretrained(model, "./storybook_model")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# 짧은 줄거리를 바탕으로 이야기를 생성해내는 프롬프트
def format_prompt(data):
    return  """
사용자 작성 내용 : """ + data + """ 

출력 구조 : 
{
  "image_prompts": [
    {
      "character_name": 등장인물 이름,
      "prompt": 등장인물 모습
    },
    ...
    ]
}


규칙 : 
- 세계관(world), 장르(genre) 를 기반으로 작성해야함
- prompt는 영어로 작성되어야만 함
- 페이지에 맞는 묘사를 해서 stable diffusion 같은 이미지 생성 모델에 처리하기 최적화 하도록 작성해야함
- python 코드 등 코드형식이 아니라 반드시 JSON형식으로 출력해야 함
- 최대한 자세하게 길게 묘사해야함

위 사용자 작성 내용, 출력 구조 그리고 규칙을 참고해서 출력으로 구체적인 묘사를 하는 출력구조로 만들어줘
"""


def write_detail_story(data):
    prompt = format_prompt(data)
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        temperature=0.5,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

