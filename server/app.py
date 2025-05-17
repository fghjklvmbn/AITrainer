from flask import Flask, request, jsonify
from utils import generate_story
from test_gen import generate_fairy_tale

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"오류": "프롬프트(prompt) 항목은 필수 입니다."}), 400
    
    result = generate_story(prompt)
    return jsonify({"result": result})


@app.route('/generate_story_test', methods=['POST'])
def generate():
    data = request.json
    # LLM 또는 템플릿 기반으로 생성
    result = generate_fairy_tale(
        world=data["world"],
        plot=data["plot"],
        main_char=data["main_character"],
        progression=data["story_progression"],
        tags=data["tags"]
    )
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
