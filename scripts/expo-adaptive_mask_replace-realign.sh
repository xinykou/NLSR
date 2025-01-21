#!/bin/bash


current_script_path=$(realpath "$0")


current_script_dir=$(dirname "$current_script_path")


parent_dir=$(dirname "$current_script_dir")  # scripts/safe_lora
sub_dir=$(dirname "$parent_dir") # ./scripts
main_dir=$(dirname "$sub_dir") # ./

cd $main_dir
# build alignment matrix
export CUDA_VISIBLE_DEVICES=1


dataset_name="sst2"
dataset_selected_all=("n1000_p0.01")
region_method=low_rank  # wanda, wandg, or low_rank

# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
sparsity_ratio=0.8
prune_rates=(0.5)
epsilon=0.2
fusion_effect=sft_to_dpo-alpha_0.9
alignment_name=expo_dpo_lora
# shellcheck disable=SC2068
for dataset_selected in ${dataset_selected_all[@]}; do
    echo "----->Running with dataset_selected=$dataset_selected"

# realign lora
for prune_rate in ${prune_rates[@]}; do
    echo "----->Running with prune_rate=$prune_rate"
    python ./safe_lora/identify_realign.py \
         --model_path ./saves/lora/sft/checkpoint-125-merged \
         --lora_path ./saves/lora/baselines/poison_ratio/aligned-finetune-${dataset_selected} \
         --aligned_path ./saves/lora/"${alignment_name}"/${fusion_effect} \
         --mask_path ./saves/lora/prune_regions/${alignment_name}-${region_method}-${sparsity_ratio}/mask_bottom_${sparsity_ratio}.pt \
         --sparsity_ratio "${sparsity_ratio}" \
         --prune_rate "${prune_rate}" \
         --epsilon ${epsilon} \
         --realign_type adaptive_mask_replace \
         --output_path ./saves/lora/realign/expo-adaptive_mask_replace-safe_lora/${dataset_name}-${alignment_name}-${dataset_selected}-${region_method}


done

done