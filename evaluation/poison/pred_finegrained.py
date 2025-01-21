import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel

os.environ["HTTP_PROXY"] = "http://127.0.0.1:27999"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:27999"
all_types = {
             "animal_abuse": 0,
             "child_abuse": 0,
             "controversial_topics,politics": 0,
             "discrimination,stereotype,injustice": 0,
             "drug_abuse,weapons,banned_substance": 0,
             "financial_crime,property_crime,theft": 0,
             "hate_speech,offensive_language": 0,
             "misinformation_regarding_ethics,laws_and_safety": 0,
             "non_violent_unethical_behavior": 0,
             "privacy_violation": 0,
             "self_harm": 0,
             "sexually_explicit,adult_content": 0,
             "terrorism,organized_crime": 0,
             "violence,aiding_and_abetting,incitement": 0
             }

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder",
                    default="/home/yx/project_v2/saves/lora/sft/checkpoint-8000-merged")
parser.add_argument("--lora_folder",
                    default="")   # ./saves/lora/sft/checkpoint-8000
parser.add_argument("--lora_folder2", default="")
parser.add_argument("--instruction_path",
                    default='BeaverTails')
parser.add_argument("--output_path",
                    default='./results/lora/sft/withorg.json')
parser.add_argument("--cache_dir", default="./cache")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_num_of_each_type", type=int, default=100)


args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
if args.output_path == "":
    raise ValueError("Please provide an output path")

output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

instruction_lst = {k: [] for k in all_types.keys()}
input_data_lst = {k: [] for k in all_types.keys()}
if "BeaverTails" in args.instruction_path:
    from datasets import load_dataset

    dataset = load_dataset("PKU-Alignment/BeaverTails", cache_dir='./data/cache')

    index = 0
    for example in dataset["30k_test"]:
        if not example["is_safe"]:
            instance = {}
            unsafe_topics = [k for k, v in example["category"].items() if v]
            if len(unsafe_topics) > 0:
                for k in unsafe_topics:
                    if all_types[k] < args.max_num_of_each_type:
                        instruction_lst[k].append(example["prompt"])
                        instance["prompt"] = example["prompt"]
                        input_data_lst[k].append(instance)
                        all_types[k] += 1 if all_types[k] < args.max_num_of_each_type else 0
                    else:
                        continue
        else:
            continue
    print("all types:", all_types)
else:
    raise ValueError("Please provide a valid instruction path")


tokenizer = AutoTokenizer.from_pretrained(args.model_folder, use_fast=True)
# tokenizer = AutoTokenizer.from_pretrained("/home/yx/project_v2/saves/lora/sft/checkpoint-8000", use_fast=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(args.model_folder, torch_dtype=torch.bfloat16, device_map="auto")

if args.lora_folder != "" or args.lora_folder2 != "":
    if args.lora_folder != "":
        print("Recover the first LoRA weights..")
        model = PeftModel.from_pretrained(
            model,
            args.lora_folder,
            torch_dtype=torch.bfloat16,
        )
        # model = model.merge_and_unload()

    if args.lora_folder2 != "":
        model = model.merge_and_unload()
        print("Recover the second LoRA weights..")
        model = PeftModel.from_pretrained(
            model,
            args.lora_folder2,
            torch_dtype=torch.bfloat16,
        )
        model = model.merge_and_unload()
        print(model)
else:
    print("No LoRA weights loaded !!!")

model.eval()


def query(instructions, batch_size=4):
    model_name = args.model_folder.lower()
    prompts = [
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
        for instruction in instructions]

    input_dict = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = input_dict['input_ids'].to(device)
    attention_mask = input_dict['attention_mask'].to(device)
    with torch.no_grad():
        generation_output = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            # top_p=1,
            # temperature=1.0,  # greedy decoding
            do_sample=False,  # greedy decoding
            num_beams=1,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in generation_output]
    results = []
    errors_num = 0
    for pro, output in zip(prompts, outputs):
        try:
            res = output.split("### Response:")[1].strip()
            if "!!!" in res:
                res = res.split("!!!")[0]
            results.append(res)
        except:
            print("Error in decoding the output")
            errors_num += 1
            results.append(output)

    return results, errors_num


batch_size = args.batch_size
pred_lst = []
print("all data size:", len(instruction_lst))

for k in instruction_lst.keys():
    instruction_lst_one_topics = instruction_lst[k]
    invalid_num = 0
    for i in tqdm(range(0, len(instruction_lst_one_topics), batch_size), desc="Generating responses"):
        batch_instructions = instruction_lst_one_topics[i:i + batch_size]
        pred, error_num = query(batch_instructions, batch_size=batch_size)
        pred_lst.extend(pred)
        invalid_num += error_num

    output_lst = []
    for input_dic, pred in zip(input_data_lst[k], pred_lst):
        input_dic['response'] = pred
        output_lst.append(input_dic)

    out_path = args.output_path.replace(".json", f"-{k}.json")
    with open(out_path, 'w') as f:
        json.dump(output_lst, f, indent=4)

    print("invalid_num:", invalid_num)


