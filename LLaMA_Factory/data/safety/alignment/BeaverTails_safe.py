from datasets import load_dataset
import os
import json
# 设置HTTP代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:27999"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:27999"


data_size = 2000

if data_size == 2000:
    output_json = f'BeaverTails_safe-2k.json'
else:
    raise ValueError("data_size must be 2000")

dataset =load_dataset("PKU-Alignment/BeaverTails", cache_dir="../../../../data/cache")

index = 0
list_data_dict =[]
for example in dataset["30k_train"]:
    if example["is_safe"] and index < 2000:
        instance = {}
        instance["instruction"] = example["prompt"]
        instance["input"] = ""
        instance["output"] = example["response"]
        list_data_dict += [instance]
        index += 1

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(list_data_dict, f, indent=4)