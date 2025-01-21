import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from model_wrapper import prune_wandg, prune_preference_wandg
from prune import get_mask, prune_wanda, prune_random
from model_wrapper_low import make_low_rank
from modeling_llama import LlamaForCausalLM_with_preference_loss

def main():
    parser = argparse.ArgumentParser(description="Identify neurons or ranks")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../saves/lora/sft/checkpoint-125-merged",
        help="Model name",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="../saves/lora/dpo",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../LLaMA_Factory/data/safety/prune_regions/dpo-safety_regions-filtered.json",
        help="Path to the safety data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../saves/lora/prune_regions/wandg-dpo-safety_regions-",
        help="Output directory",
    )
    parser.add_argument(
        "--nsamples", type=int,
        default=2000,
        help="Number of safety samples."
    )
    parser.add_argument("--seqlen",
                        type=int,
                        default=1024,
                        help="Sequence length")
    parser.add_argument(
        "--prune_method",
        type=str,
        default="wanda",
        help="Pruning methods (wandg, wanda)",
    )
    parser.add_argument(
        "--sparsity_ratio", type=float, default=0.01, help="Sparsity level"
    )
    parser.add_argument(
        "--rank", type=int, default=10, help="Rank for low rank approximation"
    )
    parser.add_argument("--niter", type=int, default=20, help="Number of iterations for low rank approximation")
    parser.add_argument(
        "--top_remove",
        action="store_true",
        help="Whether to remove the top neurons or ranks",
    )
    parser.add_argument(
        "--dump_U",
        action="store_true",
        help="Whether to dump the projection matrix",
    )
    parser.add_argument(
        "--entangle_prompt_feat",
        dest="disentangle",
        action="store_false",
        help="entangle the prompt and response when computing the wanda score",
    )

    parser.add_argument(
        "--prune_part",
        action="store_true",
        help="whether to only prune the layer with lower jaccard index",
    )
    parser.add_argument("--use_diff", action="store_true")
    parser.add_argument("--neg_prune", action="store_true",
                        help="Whether to prune the negative part of the weight matrix, default is removed from the bottom")
    parser.add_argument("--recover_from_base", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--dump_wanda_score", action="store_true", help="Whether to dump wanda scores."
    )
    parser.add_argument(
        "--prune_data",
        type=str,
        choices=[
            "wikitext",
            "alpaca",
            "alpaca_cleaned",
            "alpaca_cleaned_no_safety",
            "align",
            "align_short",
            "misalign",
            "align_misalign",
            "misalign_align",
            "align_short_misalign",
            "none",
        ],
        default="align",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Path to the data to be pruned",
    )
    parser.add_argument(
        "--use_variant",
        action="store_true",
        help="whether to use the wanda variant described in the appendix",
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        default=None,
        help="Path to save the pruned model weight mask.",
    )

    args = parser.parse_args()
    print(args)

    if args.prune_method == "preference_wandg":
        model = LlamaForCausalLM_with_preference_loss.from_pretrained(args.model_path,
                                                                      torch_dtype=torch.bfloat16,
                                                                      device_map="auto")
    else:
        if args.prune_method == "low_rank":
            model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                         torch_dtype=torch.bfloat16,
                                                         device_map="cuda:0")
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                         torch_dtype=torch.bfloat16,
                                                         device_map="auto")

    ## model = model.merge_and_unload()
    if args.lora_path:
        model = PeftModel.from_pretrained(model,
                                          args.lora_path,
                                          torch_dtype=torch.bfloat16)

    # for name, module in model.named_modules():
    #     if ('lora_A' in name or 'lora_B' in name) and 'default' in name:
    #         print(name)
    #         print(module)
    # wait = input("PRESS ENTER TO CONTINUE.")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    try:
        device = model.hf_device_map["lm_head"]
    except:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wandg":
            prune_wandg(
                args,
                model,
                tokenizer,
                device=device,
                prune_data=args.prune_data,
            )
        elif args.prune_method == "wanda":
            prune_wanda(
                args,
                model,
                tokenizer,
                device=device,
                prune_data=args.prune_data,
            )
        elif args.prune_method == "low_rank":
            device_nums = torch.cuda.device_count()
            if device_nums > 1:
                device = torch.device("cuda:1")
            else:
                raise ValueError("Low rank approximation requires at least 2 GPUs")
            make_low_rank(
                args,
                model,
                tokenizer,
                device=device,
                prune_data=args.prune_data,
            )
        elif args.prune_method == "preference_wandg":
            prune_preference_wandg(
                args,
                model,
                tokenizer,
                device=device,
                prune_data=args.prune_data,
            )
        elif args.prune_method == "random":
            prune_random(
                args,
                model,
                tokenizer,
                device=device,
                prune_data=args.prune_data,
            )

    else:
        raise ValueError("Sparsity ratio should be greater than 0")
    print("pruning done")

    if args.save_mask:
        if args.prune_method == "low_rank":
            low_rank_folder = args.output_dir
            if not os.path.exists(low_rank_folder):
                os.makedirs(low_rank_folder)
            if args.neg_prune:
                save_low_rank_path = os.path.join(
                    low_rank_folder, f"rank_top_{args.sparsity_ratio}"
                )
            else:
                save_low_rank_path = os.path.join(
                    low_rank_folder, f"rank_bottom_{args.sparsity_ratio}"
                )
            model.save_pretrained(save_low_rank_path, safe_serialization=False)

        else:
            mask = get_mask(model, args.neg_prune)
            mask_folder = args.output_dir
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)
            if args.neg_prune:
                save_mask_path = os.path.join(
                    mask_folder, f"mask_top_{args.sparsity_ratio}.pt"
                )
            else:
                save_mask_path = os.path.join(
                    mask_folder, f"mask_bottom_{args.sparsity_ratio}.pt"
                )
            torch.save(mask, save_mask_path)
            print(f"Saved weight mask to {save_mask_path}")


if __name__ == "__main__":
    main()