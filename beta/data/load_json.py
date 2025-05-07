import os
import json
from datasets import Dataset

# JSON 파일을 로드하고 변환하는 함수
def load_and_convert_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # JSON 데이터가 리스트 형식인지 확인
        if isinstance(data, list):
            processed_data = []
            for item in data:
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output_text = item.get("output", "")
                
                # 리스트일 경우 첫 번째 항목을 사용
                instruction = instruction[0] if isinstance(instruction, list) else instruction
                input_text = input_text[0] if isinstance(input_text, list) else input_text
                output_text = output_text[0] if isinstance(output_text, list) else output_text
                
                processed_data.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text
                })
            return processed_data
        else:
            raise ValueError(f"JSON 데이터는 리스트 형식이어야 합니다: {file_path}")
    
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"JSON 파일을 디코딩하는 중 오류가 발생했습니다: {file_path}")
        raise

# 폴더 내 모든 JSON 파일을 불러오고 결합하는 함수
def prepare_training_data_from_folder(folder_path):
    all_data = []
    
    # 폴더 내 모든 .json 파일 찾기
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            print(f"로드 중: {file_path}")
            file_data = load_and_convert_json(file_path)
            all_data.extend(file_data)
    
    # 전체 데이터를 Dataset 객체로 변환
    return Dataset.from_list(all_data)