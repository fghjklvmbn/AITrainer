from flask import Flask, request, jsonify
from utils import generate_story
from server.utils_test import load_model_and_tokenizer, generate_story

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"오류": "프롬프트(prompt) 항목은 필수 입니다."}), 400
    
    result = generate_story(prompt)
    return jsonify({"result": result})


model, tokenizer = load_model_and_tokenizer()

@app.route('/generate_story_test', methods=['POST'])
def generate():
    data = request.get_json()

    required_keys = ["world", "plot", "main_character", "story_progression", "tags"]
    if not all(key in data for key in required_keys):
        return jsonify({"error": "Missing required keys"}), 400

    try:
        result = generate_story(data, model, tokenizer)
        return jsonify(result)
    except Exception as e:
        return jsonify({"오류 ": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
