#!/bin/bash
# This script is used to evaluate the worst-case performance of a robust RL agent

models_dir="models_robust_large_copy"

seeds=(12345 23456 34567 45678 56789)

for seed in "${seeds[@]}"; do
    echo "Evaluating models with seed $seed"
    for extraction_method in "si-g" "alergia"; do
        mkdir -p "results_robust_large_3/${extraction_method}_seed${seed}"
    done

    for extraction_method in "si-g" "alergia"; do
        for model in $(ls $models_dir); do
            timeout 7200 prerequisites/venv/bin/python3 robust_pomdps_rl.py --project-path "$models_dir/$model" \
                --batched-vec-storm \
                --geometric-batched-vec-storm \
                --extraction-method $extraction_method \
                --seed $seed \
                > "results_robust_large_3/${extraction_method}_seed${seed}/${model}_output.txt" 2>&1
            echo "Finished evaluating $model with extraction method $extraction_method and seed $seed"
        done
    done
done
