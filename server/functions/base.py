import torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

# 처음엔 없음
model = None
tokenizer = None

# 로드할때만 적용됨
def load_model():
    global model, tokenizer

    if model is None or tokenizer is None:
        base_model_path = "Qwen/Qwen3-1.7B"
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda"
            # torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        )

        # Adapter Model 로드
        # model = PeftModel.from_pretrained(model, "./storybook_model")

        device = "cuda"
        # torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)

    return model, tokenizer


# 템플릿 로드 함수
def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# 사용자 프롬프트 템플릿별 로드
def format_prompt(data, templete):
    templete = load_prompt_template(templete)
    prompt = """사용자 작성내용 : """ + data + templete
    return prompt

def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>\n*", "", text, flags=re.DOTALL)

def generate(data, templete):
    model, tokenizer = load_model()
    prompt = format_prompt(data, templete)
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
    response = strip_think_tags(tokenizer.decode(output_ids, skip_special_tokens=True).strip())
    return response

