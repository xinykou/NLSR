#!/bin/bash


current_script_path=$(realpath "$0")


current_script_dir=$(dirname "$current_script_path")


parent_dir=$(dirname "$current_script_dir")  # scripts/ours
sub_dir=$(dirname "$parent_dir") # ./scripts
main_dir=$(dirname "$sub_dir") # ./

cd "$main_dir"


dataset_name=sst2
alignment_name=expo_dpo_lora
region_method=low_rank
data_selected_all=("n1000_p0.01" "n1000_p0.05" "n1000_p0.1" "n1000_p0.2" "n1000_p0.3")
model_path=./saves/lora/realign/expo-adaptive_mask_replace-safe_lora
#

export CUDA_VISIBLE_DEVICES=1

# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
sparsity_ratio=0.8
prune_rates=(0.5)
epsilon=0.2

for data_selected in "${data_selected_all[@]}"; do
    echo "-----> Running with data_selected=$data_selected"

    for prune_rate in "${prune_rates[@]}"; do
        echo "------> Running with prune_rate=$prune_rate"

        python ./evaluation/downstream_task/sst2_eval.py \
              --model_folder ./saves/lora/sft/checkpoint-125-merged \
              --lora_folder ${model_path}/${dataset_name}-${alignment_name}-${data_selected}-${region_method}/sparsity_ratio_"${sparsity_ratio}"_prune_rate_"${prune_rate}"_epsilon_"${epsilon}" \
              --start 0 \
              --end 1000 \
              --output_path ./results/lora/realign/expo-adaptive_mask_replace-safe_lora/${dataset_name}-${alignment_name}-${data_selected}-${region_method}/sparsity_ratio_"${sparsity_ratio}"_prune_rate_"${prune_rate}"_epsilon_"${epsilon}"-downstream.json \
              --add

    done
done



