import argparse
import torch
import os
from safetensors.torch import load_file, save_file

def main():
    parser = argparse.ArgumentParser("layer_similarity for region identification methods: wanda, wandg and low_rank")
    parser.add_argument("--first_lora_path", default="")
    parser.add_argument("--second_lora_path", default="")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--tau", default=None, type=float)
    parser.add_argument("--prune_rate", default=None, type=float)
    parser.add_argument("--sparsity_ratio", default=0.1, type=float)
    parser.add_argument("--watch_module_name", default="mlp.down_proj.lora_A", type=str)
    parser.add_argument("--step", default=3, type=int)
    args = parser.parse_args()

    print(args)
    region_similarity = []
    first_lora_weights = load_file(os.path.join(args.first_lora_path, "adapter_model.safetensors"))
    second_lora_weights = load_file(os.path.join(args.second_lora_path, "adapter_model.safetensors"))
    watch_module_name = args.watch_module_name
    for module_name, module in first_lora_weights.items():
        if watch_module_name in module_name:
            total_num = module.numel()
            same_num = torch.sum(module == second_lora_weights[module_name]).item()
            value = round((same_num / total_num), 3)
            region_similarity.append(value)

    # record by steps
    step_similarity = []
    for i in range(0, len(region_similarity), args.step):
       step_similarity.append(region_similarity[i])
    txt_name = f"region_similarity-sparsity_ratio_{args.sparsity_ratio}_prune_rate_{args.prune_rate}_{watch_module_name}.txt"
    if os.path.exists(args.output_path) is False:
        os.makedirs(args.output_path)
    txt_path = os.path.join(args.output_path, txt_name)
    txt_file = open(txt_path, "w")
    txt_file.write(" ".join([str(x) for x in step_similarity]))
    txt_file.close()


if __name__ == "__main__":
    main()