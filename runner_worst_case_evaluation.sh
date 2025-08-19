#!/bin/bash
# This script is used to evaluate the worst-case performance of a robust RL agent

source prerequisites/venv/bin/activate

models_dir="subset_evaluation"


for model in $(ls $models_dir); do
    echo "Evaluating model: $model"
    timeout 7200 python3 robust_pomdps_rl.py --project-path "$models_dir/$model" \
        --batched-vec-storm \
        --geometric-batched-vec-storm
    
    echo "Evaluating model with shrink and perturb: $model"
    timeout 7200 python3 robust_pomdps_rl.py --project-path "$models_dir/$model" \
        --batched-vec-storm \
        --geometric-batched-vec-storm \
        --shrink-and-perturb-externally

done
