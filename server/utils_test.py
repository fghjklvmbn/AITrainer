from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL_NAME = "your_finetuned_model_path_or_hub_name"

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return model, tokenizer

def format_prompt(data):
    return (
        f"입력:\n"
        f"세계관: {data['world']}\n"
        f"줄거리: {data['plot']}\n"
        f"주인공: {data['main_character']}\n"
        f"진행: {data['story_progression']}\n"
        f"태그: {', '.join(data['tags'])}\n\n"
        f"출력:"
    )

def generate_story(data, model, tokenizer):
    prompt = format_prompt(data)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.1
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # '출력:' 이후만 파싱
    output_section = decoded.split("출력:")[-1].strip()
    
    # 안전을 위해 JSON 추출 시 오류 처리 (문자열로 시작하는 경우만 처리)
    import json
    try:
        output_json = json.loads(output_section)
    except json.JSONDecodeError:
        raise ValueError("모델 출력이 유효한 JSON 형식이 아닙니다.")
    
    return output_json