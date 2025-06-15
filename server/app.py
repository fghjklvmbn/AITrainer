from flask import Flask, request, jsonify
from functions.Chatbot import chat
from functions.base import generate
from functions.translate import translate_text
import os, json

# 현재 파일 위치
path = os.getcwd()
print(path)

app = Flask(__name__)

# 전체적인 이야기 생성
@app.route("/ai/StoryCreate/generate", methods=["POST"])
def generate_story():
    data = request.get_json()
    prompt = data.get("prompt", "")
    templete = path + "\\functions\\templete\\createstory.txt"
    
    if not prompt:
        return jsonify({"오류": "프롬프트(prompt) 항목은 필수 입니다."}), 400
    
    result = generate(prompt, templete)
    first_parse = json.loads(result)     # result: 문자열
    return jsonify(first_parse)

# 이야기 생성(5페이지)
@app.route("/ai/StoryCreate/write", methods=["POST"])
def write_detail_story():
    prompt = request.get_json()

    # JSON을 예쁘게 포맷해서 문자열로 만들기
    pretty_json = json.dumps(prompt, ensure_ascii=False, indent=4)

    # 삼중 따옴표로 감싸기
    wrapped_json = f'"""{pretty_json}"""'
    
    # 전용 템플릿 전달
    templete = path + "\\functions\\templete\\detail.txt"
    
    if not prompt:
        return jsonify({"오류": "프롬프트(prompt) 항목은 필수 입니다."}), 400
    
    # write_story 함수를 호출하여 이야기를 생성한다.
    result = generate(wrapped_json, templete)
    first_parse = json.loads(result)     # result: 문자열
    return jsonify(first_parse)

# 이야기 수정
@app.route("/ai/StoryCreate/modify", methods=["POST"])
def modify_story():
    prompt = request.get_json()

    # JSON을 예쁘게 포맷해서 문자열로 만들기
    pretty_json = json.dumps(prompt, ensure_ascii=False, indent=4)

    # 삼중 따옴표로 감싸기
    wrapped_json = f'"""{pretty_json}"""'
    
    # 전용 템플릿 전달
    templete = path + "\\functions\\templete\\modify.txt"
    
    if not prompt:
        return jsonify({"오류": "프롬프트(prompt) 항목은 필수 입니다."}), 400
    
    result = generate(wrapped_json, templete)
    first_parse = json.loads(result)     # result: 문자열
    return jsonify(first_parse)

# 등장인물 자세한 정보 생성
@app.route("/ai/StoryCreate/charspec", methods=["POST"])
def character_spec():
    prompt = request.get_json()

    # JSON을 예쁘게 포맷해서 문자열로 만들기
    pretty_json = json.dumps(prompt, ensure_ascii=False, indent=4)

    # 삼중 따옴표로 감싸기
    wrapped_json = f'"""{pretty_json}"""'

    # 전용 템플릿 전달
    templete = path + "\\functions\\templete\\charspec.txt"

    if not prompt:
        return jsonify({"오류": "프롬프트(prompt) 항목은 필수 입니다."}), 400
    
    result = generate(wrapped_json, templete)
    first_parse = json.loads(result)     # result: 문자열
    return jsonify(first_parse)

# 등장인물 자세한 정보 생성
@app.route("/ai/StoryCreate/artprompt/", methods=["POST"])
def artprompt():
    prompt = request.get_json()

    # JSON을 예쁘게 포맷해서 문자열로 만들기
    pretty_json = json.dumps(prompt, ensure_ascii=False, indent=4)

    # 삼중 따옴표로 감싸기
    wrapped_json = f'"""{pretty_json}"""'

    # 전용 템플릿 전달
    templete = path + "\\functions\\templete\\charspec.txt"

    if not prompt:
        return jsonify({"오류": "프롬프트(prompt) 항목은 필수 입니다."}), 400
    
    result = generate(wrapped_json, templete)
    first_parse = json.loads(result)     # result: 문자열
    return jsonify(first_parse)

# 챗봇 대화 기능
@app.route("/ai/chat", methods=["POST"])
def chatting():
    data = request.get_json()
    user_input = data.get('input', '')
    
    if not user_input:
        return jsonify({'error': 'Empty input'}), 400

    try:
        response = chat(user_input)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 번역 기능
@app.route("/ai/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "")
    target_language = data.get("target_language", "en")

    if not text:
        return jsonify({"error": "번역할 텍스트가 필요합니다."}), 400

    translated_text = translate_text(text, target_language)
    return jsonify({"translated_text": translated_text})

@app.route("/ai/StoryCreate/", methods=["GET"])
def index():
    return jsonify({"message": "정상작동"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
