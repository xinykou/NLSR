import argparse
import torch
from peft import load_peft_weights
from peft.utils import WEIGHTS_NAME
import os


def main():
    parser = argparse.ArgumentParser(description="sparsity ranks")
    parser.add_argument(
        "--rank_path",
        type=str,
        default="../saves/lora/sft/checkpoint-125-merged",
        help="Model name",
    )

    parser.add_argument(
        "--sparsity_ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../saves/lora/prune_regions/wandg-dpo-safety_regions-",
        help="Output directory",
    )
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    peft_weights = torch.load(os.path.join(args.rank_path, WEIGHTS_NAME), map_location=device)

    binary_masks = {}
    for name, weight in peft_weights.items():
        W = weight.data
        W_metric = torch.abs(W)
        values, index = torch.sort(W_metric.flatten())
        thresh = values[int(len(values) * args.sparsity_ratio)]
        print(f"Layer: {name}    Threshold: {thresh}")
        print(W_metric.flatten().cpu().mean())
        if thresh == 0:
            frac_zero = (W_metric == 0).sum().item() / W_metric.numel()
            W_mask = (W_metric == 0) * (torch.rand_like(W_metric) < (args.sparsity_ratio / frac_zero))
        else:
            W_mask = W_metric <= thresh

        new_name = name.replace("weight", "default")
        uility_W_mask = ~W_mask
        binary_masks[new_name] = uility_W_mask
        ratio = uility_W_mask.sum().item() / uility_W_mask.numel()
        print(f"Layer: {name}   Safety region ratio: {ratio}")

    print(f"Saving pruned weights to {args.output_dir}")
    torch.save(binary_masks, os.path.join(args.output_dir, f'mask_bottom_{args.sparsity_ratio}.pt'))


if __name__ == "__main__":
    main()