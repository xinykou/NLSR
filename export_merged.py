import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from peft import PeftModel
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--org_model_path", type=str, default="/media/2/yx/model_cache/Meta-Llama-3-8B")
    parser.add_argument("--lora_path", type=str, default="")

    parser.add_argument("--save_path", type=str, default="/home/yx/project_v2/saves/lora/sft/checkpoint-8000-withours")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.org_model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map='auto')

    tokenizers = AutoTokenizer.from_pretrained(args.org_model_path)

    model = PeftModel.from_pretrained(
        model,
        args.lora_path,
        torch_dtype=torch.bfloat16
    )

    model = model.merge_and_unload()

    model.save_pretrained(args.save_path)
    tokenizers.save_pretrained(args.save_path)

    print("merged and saved!")


if __name__ == '__main__':
    main()
