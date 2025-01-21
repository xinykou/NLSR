import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel


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
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=10)

args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
if args.output_path == "":
    raise ValueError("Please provide an output path")

output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

instruction_lst = []
input_data_lst = []
if "BeaverTails" in args.instruction_path:
    from datasets import load_dataset

    dataset = load_dataset("PKU-Alignment/BeaverTails", cache_dir='./data/cache')
    index = 0
    end_loc = 1000 if args.end == 1000 else args.end
    start_loc = 0 if args.start == 0 else args.start
    for example in dataset["30k_test"]:
        if not example["is_safe"]:
            if start_loc <= index < end_loc:
                instance = {}
                instance["prompt"] = example["prompt"]
                instruction_lst += [example["prompt"]]
                input_data_lst += [instance]
                index += 1
            else:
                index += 1
        else:
            continue
if "harmbench" in args.instruction_path:
    with open(args.instruction_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        for example in dataset:
            instance = {}
            instance["prompt"] = example["instruction"]
            instruction_lst.append(example["instruction"])
            input_data_lst.append(instance)

else:
    if args.instruction_path.endswith(".json"):
        with open(args.instruction_path, 'r', encoding='utf-8') as f:
            input_json = json.load(f)
            for example in input_json:
                instance = {}
                instance["prompt"] = example['prompt']
                instruction_lst.append(example['prompt'])
                input_data_lst.append(instance)
    elif args.instruction_path.endswith(".jsonl"):
        with open(args.instruction_path, 'r', encoding='utf-8') as f:
            for line in f:
                instance = {}
                example = json.loads(line)
                instance["prompt"] = example['prompt']
                instruction_lst.append(example['prompt'])
                input_data_lst.append(instance)


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

invalid_num = 0
for i in tqdm(range(0, len(instruction_lst), batch_size), desc="Generating responses"):
    batch_instructions = instruction_lst[i:i + batch_size]
    pred, error_num = query(batch_instructions, batch_size=batch_size)
    pred_lst.extend(pred)
    invalid_num += error_num

output_lst = []
for input_dic, pred in zip(input_data_lst, pred_lst):
    input_dic['response'] = pred
    output_lst.append(input_dic)

with open(args.output_path, 'w') as f:
    json.dump(output_lst, f, indent=4)

print("invalid_num:", invalid_num)


