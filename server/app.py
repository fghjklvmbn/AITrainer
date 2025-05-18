from flask import Flask, request, jsonify
from chatbot import Chatbot
# from utils import generate_story

app = Flask(__name__)
chatbot = Chatbot()  # 챗봇 인스턴스 생성

# api url 설정
# @app.route("/ai/StoryCreate/fc", methods=["POST"])
# @app.route("/ai/StoryCreate/write", methods=["POST"])
# @app.route("/ai/StoryCreate/modify", methods=["POST"])
# @app.route("/ai/StoryCreate/character_sp", methods=["POST"])

# 기타 api url 설정 : 대화(chat) 기능
@app.route("/ai/chat", methods=["POST"])
@app.route("/ai/reset", methods=["POST"])

# def fc_story():
#     data = request.get_json()
#     prompt = data.get("prompt", "")
    
#     if not prompt:
#         return jsonify({"오류": "프롬프트(prompt) 항목은 필수 입니다."}), 400
    
#     result = generate_story(prompt)
#     return jsonify({"result": result})


# def write_story():
#     data = request.get_json()
#     prompt = data.get("prompt", "")
    
#     if not prompt:
#         return jsonify({"오류": "프롬프트(prompt) 항목은 필수 입니다."}), 400
    
#     result = generate_story(prompt)
#     return jsonify({"result": result})


# def modify_story():
#     data = request.get_json()
#     prompt = data.get("prompt", "")
    
#     if not prompt:
#         return jsonify({"오류": "프롬프트(prompt) 항목은 필수 입니다."}), 400
    
#     result = generate_story(prompt)
#     return jsonify({"result": result})

# def character_sp():
#     data = request.get_json()
#     prompt = data.get("prompt", "")
    
#     if not prompt:
#         return jsonify({"오류": "프롬프트(prompt) 항목은 필수 입니다."}), 400
    
#     result = generate_story(prompt)
#     return jsonify({"result": result})


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
