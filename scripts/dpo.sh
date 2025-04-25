#!/bin/bash

current_script_path=$(realpath "$0")
current_script_dir=$(dirname "$current_script_path")

parent_dir=$(dirname "$current_script_dir")  # scripts/alignment
sub_dir=$(dirname "$parent_dir") # scripts
main_dir=$(dirname "$sub_dir") # ./

cd $main_dir

export WANDB_PROJECT="assessing_safety"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$main_dir

python main.py train config/DPO.yaml