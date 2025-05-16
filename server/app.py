from flask import Flask, request, jsonify
from utils import generate_story

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"오류": "프롬프트(prompt) 항목은 필수 입니다."}), 400
    
    result = generate_story(prompt)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
