# app.py
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

app = Flask(__name__)

# Load LoRA-adapted model
base_model_name = "your-base-model"  # e.g., "gpt2" or "tiiuae/falcon-7b"
peft_model_path = "./lora-checkpoint"  # directory where adapter_config.json exists

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16).to("cuda")
model = PeftModel.from_pretrained(model, peft_model_path)
model.eval()

@app.route('/generate_story', methods=['POST'])
def generate_story():
    data = request.json

    # Validate input keys
    required_keys = {"world", "plot", "main_character", "story_progression", "tags"}
    if not required_keys.issubset(set(data.keys())):
        return jsonify({"error": "Missing required keys"}), 400

    prompt = (
        f"입력: \n"
        f'{{\n'
        f'  "world": "{data["world"]}",\n'
        f'  "plot": "{data["plot"]}",\n'
        f'  "main_character": "{data["main_character"]}",\n'
        f'  "story_progression": "{data["story_progression"]}",\n'
        f'  "tags": {data["tags"]}\n'
        f'}}\n\n'
        f"출력: \n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response_text = decoded.split("출력:")[-1].strip()

    # Try to parse output as JSON (if possible)
    try:
        import json
        story_json = json.loads(response_text)
        return jsonify(story_json)
    except json.JSONDecodeError:
        return jsonify({"output": response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)