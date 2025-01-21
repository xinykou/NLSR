import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel, load_peft_weights
import os
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--org_model_path', type=str,
                        help='Path to the original model')
    parser.add_argument('--weak_model_path', type=str,
                        default='./saves/lora/sft/checkpoint-125-merged'
                        )
    parser.add_argument('--moderate_model_path', type=str,
                        default='./saves/lora/expo-dpo')
    parser.add_argument('--alpha', type=float,
                        default=0.5)
    parser.add_argument('--save_path', type=str,
                        default='')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA weights')

    # parser.add_argument('--mask_path', type=str,
    #                     default='./saves/lora/prune_regions/wanda-dpo/mask_bottom_0.01.pt',
    #                     help='Path to the mask file')

    args = parser.parse_args()

    print(args)

    if args.use_lora:
        org_model = AutoModelForCausalLM.from_pretrained(
            args.org_model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto' if torch.cuda.is_available() else 'cpu',
            trust_remote_code=True,
        )
        if args.weak_model_path is not None and args.moderate_model_path is not None:
            org_model_copy = copy.deepcopy(org_model)
            weak_model = PeftModel.from_pretrained(
                org_model_copy,
                args.weak_model_path,
                torch_dtype=torch.bfloat16,
            )
            print('LORA weak model to be merged')
            weak_model = weak_model.merge_and_unload()
            print('LoRA weak model loaded')

            org_model_copy_copy = copy.deepcopy(org_model)
            moderate_model = PeftModel.from_pretrained(
                org_model_copy_copy,
                args.moderate_model_path,
                torch_dtype=torch.bfloat16,
            )  # device_map='auto' if torch.cuda.is_available() else 'cpu')
            print('LORA moderate model to be merged')
            moderate_model = moderate_model.merge_and_unload()
            print('LORA moderate model loaded')
        else:
            raise ValueError('Please provide weak and moderate model paths')

    else:
        weak_model = AutoModelForCausalLM.from_pretrained(
            args.weak_model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto' if torch.cuda.is_available() else 'cpu',
        )
        print('Weak model loaded')
        moderate_model = PeftModel.from_pretrained(
            weak_model,
            args.moderate_model_path,
            torch_dtype=torch.bfloat16,
        )
        print('Moderate model loaded')

    total = len(moderate_model.state_dict())
    for name, moderate_model_param in tqdm(moderate_model.named_parameters(), total=total, desc='expo...'):
        if 'lora' in name:
            moderate_model_param.data = args.alpha * moderate_model_param.data + args.alpha * moderate_model_param.data
            # print(f"{name} scaled.")

    # moderate_model = moderate_model.merge_and_unload()
    moderate_model.save_pretrained(args.save_path, safe_serialization=False)
    toker = AutoTokenizer.from_pretrained(args.moderate_model_path, trust_remote_code=True)
    toker.save_pretrained(args.save_path)
    print('expo done and saved: ', args.save_path)


if __name__ == '__main__':
    main()
