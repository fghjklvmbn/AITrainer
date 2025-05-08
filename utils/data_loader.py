import os
import json
from datasets import Dataset

def load_dataset(json_dir):
    all_texts = []
    for file_name in os.listdir(json_dir):
        if file_name.endswith('.json'):
            with open(os.path.join(json_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        all_texts.append({"text": item.get("content", "")})
                elif isinstance(data, dict):
                    all_texts.append({"text": data.get("content", "")})
    dataset = Dataset.from_list(all_texts)
    return dataset