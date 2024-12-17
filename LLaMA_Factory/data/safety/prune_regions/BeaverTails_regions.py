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

    args = parser.parse_args()
    print(args)

    output_json = args.output_path

    if not os.path.exists(output_json):
        dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", cache_dir=args.data_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="auto"
                                                     )
        if args.lora_path:
            model = PeftModel.from_pretrained(model,
                                              args.lora_path,
                                              torch_dtype=torch.bfloat16, )

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        list_data_dict = []
        current_safe_size = 0
        max_safe_size = args.safety_size
        batch_size = args.batch_size

        def build_dict(prompt):
            instance = {"input": prompt, "output": ""}
            return instance

        def query(instructions, batch_size=4):

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

        for example in dataset["test"]:
            instance = build_dict(example["prompt"])
            list_data_dict.append(instance)

        for i in tqdm(range(0, len(list_data_dict), batch_size), desc="Generating responses for regions identification"):
            batch = list_data_dict[i:i + batch_size]
            outputs, error_num = query([instance["input"] for instance in batch], batch_size=batch_size)
            print(f"error_num: {error_num}")
            for instance, output in zip(batch, outputs):
                instance["output"] = output

            current_safe_size += len(batch)
            if current_safe_size >= max_safe_size:
                break

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(list_data_dict, f, indent=4)

    else:
        # filter out unsafe responses
        with open(output_json, 'r', encoding='utf-8') as f:
            list_data_dict = json.load(f)
        filtered_list_data_dict = []
        evaluator_model = QAModeration.from_pretrained(
            args.safety_evaluator_path,
            model_max_length=256,
            device_map='auto',
        )
        evaluator_model.eval()
        for instance in tqdm(list_data_dict, desc="Filtering out unsafe responses by safety evaluator"):
            predictions = evaluator_model.predict(
                question=instance["input"],
                answer=instance["output"],
                batch_size=1,
                return_bool=True,
                threshold=0.5,
            )
            if predictions["flagged"]:
                continue
            else:
                filtered_list_data_dict.append(instance)

        filtered_output_json = output_json.replace(".json", f"-filtered.json")

        print(f"Filtered out {len(list_data_dict) - len(filtered_list_data_dict)} unsafe responses")
        with open(filtered_output_json, 'w', encoding='utf-8') as f:
            json.dump(filtered_list_data_dict, f, indent=4)


if __name__ == "__main__":
    main()
