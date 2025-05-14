from datasets import load_dataset


dataset = load_dataset("json", data_files = {"train": "data/json_files/data1.json"}, split = "train")
print(dataset)