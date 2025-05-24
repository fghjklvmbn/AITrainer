from flask import Flask, request, jsonify
from server.functions.Chatbot import Chatbot
from server.functions.CreateStory import generate_story
from server.functions.WriteDetailStory import write_story
from server.functions.ModifyStory import modify_story
from server.functions.CharaterSpec import character_spec
from server.functions.translate import translate_text

app = Flask(__name__)
chatbot = Chatbot()  # 챗봇 인스턴스 생성

# api url 설정
@app.route("/ai/StoryCreate/generate", methods=["POST"])
@app.route("/ai/StoryCreate/write", methods=["POST"])
@app.route("/ai/StoryCreate/modify", methods=["POST"])
@app.route("/ai/StoryCreate/character_sp", methods=["POST"])

# (베타)기타 api url 설정 : 대화(chat) 기능
@app.route("/ai/chat", methods=["POST"])
@app.route("/ai/reset", methods=["POST"])

# 번역 기능
@app.route("/ai/translate", methods=["POST"])

# 전체적인 이야기 생성
def generate_story():
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"오류": "프롬프트(prompt) 항목은 필수 입니다."}), 400
    
    result = generate_story(prompt)
    return jsonify({"result": result})

# 부차적인 이야기 생성(ex, 추가 내용 만들기)
# 페이지수, 스토리 진행을 받아서 이야기 생성
# 만약 1
def write_story():
    data = request.get_json()

    # 프롬프트와 스토리 진행을 가져온다.
    page = data.get("page", "")
    story_progression = data.get("story_progression", "")
    
    if not page:
        return jsonify({"오류": "페이지(page) 항목은 필수 입니다."}), 400
    if not story_progression:
        return jsonify({"오류": "스토리 진행(story_progression) 항목은 필수 입니다."}), 400
    
    # write_story 함수를 호출하여 이야기를 생성한다.
    result = write_story(page, story_progression)
    return jsonify({"result": result})

# 이야기 수정
def modify_story():
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"오류": "프롬프트(prompt) 항목은 필수 입니다."}), 400
    
    result = modify_story(prompt)
    return jsonify({"result": result})

# 등장인물 자세한 정보 생성
def character_spec():
    data = request.get_json()
    character = data.get("character_list", "")
    
    if not character:
        return jsonify({"오류": "캐릭터(character) 항목은 필수 입니다."}), 400
    
    result = character_spec(character)
    return jsonify({"result": result})

# (베타) 챗봇 대화 기능
def chat():
    data = request.json
    user_input = data.get("input", "")

    if not user_input:
        return jsonify({"error": "입력이 없습니다."}), 400

    response = chatbot.generate_response(user_input)
    return jsonify({"response": response})

def reset():
    chatbot.reset()
    return jsonify({"status": "대화가 초기화되었습니다."})

def translate():
    data = request.get_json()
    text = data.get("text", "")
    target_language = data.get("target_language", "en")

    if not text:
        return jsonify({"error": "번역할 텍스트가 필요합니다."}), 400

    translated_text = translate_text(text, target_language)
    return jsonify({"translated_text": translated_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
