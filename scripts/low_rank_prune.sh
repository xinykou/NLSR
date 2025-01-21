#!/bin/bash

current_script_path=$(realpath "$0")

current_script_dir=$(dirname "$current_script_path")

parent_dir=$(dirname "$current_script_dir")  # scripts
sub_dir=$(dirname "$parent_dir") # ./

cd $sub_dir
echo "Current working directory: $sub_dir"

export CUDA_VISIBLE_DEVICES=1

prune_type=low_rank
sparsity_ratios=(0.8)
alignment_types=("expo_dpo_lora") # ("dpo" "kto" "simpo" "orpo" "expo_dpo_lora" "expo_kto_lora" "expo_simpo_lora" "expo_orpo_lora")
fusion_effects=("sft_to_dpo-alpha_0.9")



# shellcheck disable=SC2068
for sparsity_ratio in ${sparsity_ratios[@]}
do
    for alignment_name in ${alignment_types[@]}
    do
      echo "----->sparsity_ratio: ${sparsity_ratio}..."
      echo "--->Alignment type: ${alignment_name}..."

      alignment_name="${alignment_types[$i]}"
      if [[ "$alignment_name" == *"expo"* ]]; then  # if alignment_name contains "expo"
          modified_alignment_name="${alignment_name}/${fusion_effects[$i]}"
      else
          modified_alignment_name="${alignment_name}"
      fi
      python ./prune_regions/sparsity_ratio_low.py \
            --rank_path ./saves/lora/prune_regions/${alignment_name}-${prune_type}-${sparsity_ratio}/rank_bottom_${sparsity_ratio} \
            --output_dir ./saves/lora/prune_regions/${alignment_name}-${prune_type}-${sparsity_ratio} \
            --sparsity_ratio ${sparsity_ratio}
    done
done

