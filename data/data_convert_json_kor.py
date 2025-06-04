from pathlib import Path
import json
import os

# 동화 텍스트 파일 경로
file_path = "한국전래동화, 우화(링크2번).txt"
output_path = "./json_files/data_전래동화.json"

# 텍스트 불러오기
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 동화들을 빈 줄 기준으로 분리 후 한 줄로 변환
stories = [story.strip().replace("\n", " ") for story in text.split("\n\n") if story.strip()]

# JSON 배열 데이터 생성
json_data = [
    {
        "instruction": "한국 전래동화 스타일로 이야기 써줘",
        "input": "",
        "output": story
    }
    for story in stories
]

# 디렉토리 없으면 생성
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# JSON 파일로 저장 (대괄호 + 콤마 포함된 정식 JSON 배열)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"저장 완료: {output_path}")
