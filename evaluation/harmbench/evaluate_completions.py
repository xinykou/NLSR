import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
import argparse
import os
os.environ["OPENAI_API_KEY"] = "sk-"
import csv
from tqdm import tqdm 
import numpy as np

from vllm import LLM, SamplingParams

from eval_utils import compute_results_classifier, compute_results_advbench
# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    parser.add_argument("--cls_path", type=str, default='cais/HarmBench-Llama-2-13b-cls',
                        help="The name or path of the classifier for computing ASR")
    parser.add_argument("--behaviors_path", type=str, default='',
                        help="The path to the behaviors file")
    parser.add_argument("--completions_path", type=str,
                        help="The path to the completions to evaluate")
    parser.add_argument("--save_path", type=str,
                        help="The path for saving results")
    parser.add_argument("--include_advbench_metric", action="store_true",
                        help="Whether to include the AdvBench refusal metric")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="The number of tokens to evaluate")
    parser.add_argument("--generate_with_vllm", action="store_true",)
    parser.add_argument("--record_mask", action="store_true",)
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)

    # ========== initialize results ========== #
    with open(args.completions_path, 'r') as f:
        completions_list = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
    tokenizer.truncation_side = "right"

    # ========== evaluate completions ========== #
    if args.generate_with_vllm:
        cls = LLM(model=args.cls_path,
                  tensor_parallel_size=2,
                  dtype="bfloat16",
                  gpu_memory_utilization=0.9
                  )

        # cls.llm_engine.tokenizer.tokenizer.truncation_side = "left"

        cls_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    else:
        cls = AutoModelForCausalLM.from_pretrained(args.cls_path,
                                                   device_map="auto",
                                                   torch_dtype=torch.bfloat16)
        cls_params = GenerationConfig(max_new_tokens=args.max_new_tokens, num_beams=1, temperature=0.0, do_sample=False)

    results = {}
    error_num = 0
    for item_data in tqdm(completions_list, total=len(completions_list), desc="Evaluating completions"):
        current_results = compute_results_classifier(item_data, cls, cls_params)
        # todo: if args.include_advbench_metric:
        #current_results = compute_results_advbench(current_results, item_data)
        if current_results == -1:
            error_num += 1
            continue

        item_data['attacked_or_not'] = current_results
        if "default" not in results:
            results['default'] = []

        results['default'].append(current_results)

    sub_success_num_records = []
    sub_success_rate_records = []
    for i, (behavior_id, data) in enumerate(results.items()):
        sub_success_nums = np.sum(data)
        sub_success_rate = round(sub_success_nums / len(data), 3)
        sub_asr = f"{i}. {behavior_id} === average asr {sub_success_rate}"
        print(sub_asr)
        sub_success_num_records.append(sub_success_nums)
        sub_success_rate_records.append(sub_asr)

    print(f"Error number: {error_num}")

    total_success_rate = round(sum(sub_success_num_records) / len(completions_list), 3)
    aver_asr = f"Average ASR: {total_success_rate}"
    print(aver_asr)

    completions_list.extend(sub_success_rate_records)
    completions_list.append(aver_asr)
    # Make dirs to output_file if not yet exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
    with open(args.save_path, 'w') as file:
        json.dump(completions_list, file, indent=4)

    print("Results saved to", args.save_path)

    if args.record_mask:
        name = args.save_path.split("/")[-1]
        record_path = args.save_path.replace(name, "mask.txt")
        with open(record_path, 'a+') as f:
            f.write(name + ':' + aver_asr + '\n')


if __name__ == "__main__":
    main()
