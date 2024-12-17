import argparse
import json
import random

random.seed(0)

parser = argparse.ArgumentParser()
args = parser.parse_args()

from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca")
output_json = f'train.json'
output_data_lst = []
for data in dataset["train"]:
    print(data)
    item = {}
    item["instruction"] = data["instruction"]
    item["input"] = data["input"]
    item["output"] = data["output"]
    output_data_lst += [item]
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output_data_lst, f, indent=4)
