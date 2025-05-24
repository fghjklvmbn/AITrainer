from flask import Flask, request, jsonify
from stub.WriteDetail_stub import WriteDetail
from stub.FullCreate_stub import FullCreate
from stub.CharaterSpecific_stub import CharaterSpecific

app = Flask(__name__)

# api url 설정

@app.route("/ai/StoryCreate/WriteDetail", methods=["POST"])
@app.route("/ai/StoryCreate/FullCreate", methods=["POST"])
@app.route("/ai/StoryCreate/CharaterSpecific", methods=["POST"])

# 전체적인 이야기 생성
def FullCreate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    custom_tag = data.get("custom_tag", "")
    
    if not prompt:
        return jsonify({"오류": "프롬프트(prompt) 항목은 필수 입니다."}), 400
    if not custom_tag:
        return jsonify({"오류": "커스텀 태그(custom_tag) 항목은 필수 입니다."}), 400

    result = FullCreate(prompt, custom_tag)
    return jsonify(result)


def WriteDetail():
    data = request.get_json()

    # 프롬프트와 스토리 진행을 가져온다.
    createpage = data.get("page", "")
    story_progression = data.get("story_progression", "")
    
    if not createpage:
        return jsonify({"오류": "페이지(page) 항목은 필수 입니다."}), 400
    if not story_progression:
        return jsonify({"오류": "스토리 진행(story_progression) 항목은 필수 입니다."}), 400
    
    # write_story 함수를 호출하여 이야기를 생성한다.
    result = WriteDetail(createpage, story_progression)
    return jsonify(result)


# 등장인물 자세한 정보 생성
def CharaterSpecific():
    data = request.get_json()
    character_name = data.get("character_name", "")
    plot = data.get("plot", "")
    
    if not character_name:
        return jsonify({"오류": "캐릭터(character) 항목은 필수 입니다."}), 400
    if not plot:
        return jsonify({"오류": "줄거리(plot) 항목은 필수 입니다."}), 400

    result = CharaterSpecific(character_name, plot)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3001, debug=True)
