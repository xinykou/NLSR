import torch
from datasets import load_dataset
import os
import json
import argparse
from tqdm import tqdm
import sys
print(f"path: {sys.path}")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from evaluation.poison.moderation import QAModeration

# 设置HTTP代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:27999"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:27999"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str,
                        default="safety_regions.json")
    parser.add_argument("--safety_size", type=int,
                        default=20000)
    parser.add_argument("--model_path", type=str,
                        default="../../../../saves/lora/sft/checkpoint-125-merged")

    parser.add_argument("--lora_path", type=str,
                        default="../../../../saves/lora/dpo")

    parser.add_argument("--batch_size", type=int,
                        default=16)

    parser.add_argument("--safety_evaluator_path", type=str,
                        default="../../../../pretrained_model/beaver-dam-7b")

    parser.add_argument("--alignment_method", type=str,
                        default="dpo")

    parser.add_argument("--data_dir", type=str,
                         default="../../../../data/cache")

    parser.add_argument("--one_response_data_dir", type=str,
                        default="",
                        help="dataset for single response for search for the safety regions"
                        )

    args = parser.parse_args()
    print(args)

    output_json = args.output_path
    list_data_dict = []

    selected_data = []
    if args.one_response_data_dir != "":
        data_all = json.load(open(args.one_response_data_dir, 'r'))
        for data in data_all:
            selected_data.append(data["input"])

    if not os.path.exists(output_json):
        dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", cache_dir=args.data_dir)

        def build_dict(example=None):
            instance = {"input": example["prompt"]}
            if example["better_response_id"] == 0:
                instance["chosen"] = example["response_0"]
                instance["rejected"] = example["response_1"]
            else:
                instance["chosen"] = example["response_1"]
                instance["rejected"] = example["response_0"]
            return instance

        for example in dataset["test"]:
            instance = build_dict(example)
            if instance["input"] in selected_data:
                list_data_dict.append(instance)
            else:
                continue

        print(f"filter data size: {len(list_data_dict)}")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(list_data_dict, f, indent=4)

    else:
        print(f"{output_json} already exists, skip the processing.")


if __name__ == "__main__":
    main()
