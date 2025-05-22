import json

def extract_valid_json(text: str):
    """
    주어진 텍스트에서 JSON 형식으로 보이는 부분만 추출하고,
    구조 조건을 만족하는 경우에만 리턴한다.
    """
    try:
        # 첫 번째 중괄호부터 추출
        json_start = text.find("{")
        if json_start == -1:
            return None

        json_candidate = text[json_start:]
        
        # 중괄호 짝 맞추기: 유효 JSON 범위 추정
        brace_stack = []
        end_index = -1
        for i, char in enumerate(json_candidate):
            if char == '{':
                brace_stack.append('{')
            elif char == '}':
                if brace_stack:
                    brace_stack.pop()
                if not brace_stack:
                    end_index = i + 1
                    break
        if end_index == -1:
            return None

        json_str = json_candidate[:end_index]

        # JSON 파싱
        parsed = json.loads(json_str)

        # 필수 필드 확인
        required_fields = ["world", "genre", "characters", "plot", "story_progression", "tags"]
        if not all(field in parsed for field in required_fields):
            return None

        # characters 검사
        if not isinstance(parsed["characters"], list) or len(parsed["characters"]) == 0:
            return None

        for character in parsed["characters"]:
            if not all(k in character for k in ["character_name", "gender", "personality", "ability", "main_character"]):
                return None
            if not isinstance(character["main_character"], bool):
                return None

        # tags 검사
        if not isinstance(parsed["tags"], list) or len(parsed["tags"]) != 3:
            return None

        return parsed

    except Exception:
        return None