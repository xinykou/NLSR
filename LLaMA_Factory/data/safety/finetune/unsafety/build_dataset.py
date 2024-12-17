from datasets import load_dataset
import os
import json


output_json = f'BeaverTails_unsafe.json'

dataset =load_dataset("PKU-Alignment/BeaverTails")

list_data_dict =[]
for example in dataset["30k_train"]:
    if not example["is_safe"]:
        instance = {}
        instance["instruction"] = example["prompt"]
        instance["input"] = ""
        instance["output"] = example["response"]
        list_data_dict += [instance]
# list_data_dict= list_data_dict[:1000]

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(list_data_dict, f, indent=4)