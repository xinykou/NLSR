#!/bin/bash


current_script_path=$(realpath "$0")


current_script_dir=$(dirname "$current_script_path")

parent_dir=$(dirname "$current_script_dir")  # scripts
sub_dir=$(dirname "$parent_dir") # ./


cd $sub_dir

source_type=sft
target_type=dpo
alpha_all=(0.9) # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
#shellcheck disable=SC2068

echo " Optimal alpha: ${alpha}..."

 CUDA_VISIBLE_DEVICES="" python ./weak_to_strong/expo-lora.py \
   --weak_model_path ./saves/lora/${source_type}/checkpoint-125-merged \
   --moderate_model_path ./saves/lora/${target_type} \
   --alpha ${alpha} \
   --save_path ./saves/lora/expo_${target_type}_lora/${source_type}_to_${target_type}-alpha_${alpha} \



