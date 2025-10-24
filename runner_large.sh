#!/bin/bash
# This script is used to evaluate the worst-case performance of a robust RL agent

models_dir="models_large_fam_3"

seeds=(12345 23456 34567 45678 56789)

for seed in "${seeds[@]}"; do
    echo "Evaluating models with seed $seed"
    for extraction_method in "si-g" "alergia"; do
        mkdir -p "results_robust_large_fam/${extraction_method}_seed${seed}"
    done

    for extraction_method in "si-g" "alergia"; do
        for model in $(ls $models_dir); do
            # if model is not directory, skip
            if [ ! -d "$models_dir/$model" ]; then
                echo "Skipping $model as it is not a directory"
                continue
            fi
            echo "Evaluating $model with extraction method $extraction_method and seed $seed"
            timeout 3600 prerequisites/venv/bin/python3 robust_pomdps_rl.py --project-path "$models_dir/$model" \
                --batched-vec-storm \
                --geometric-batched-vec-storm \
                --extraction-method $extraction_method \
                --seed $seed \
                > "results_robust_large_fam/${extraction_method}_seed${seed}/${model}_output.txt" 2>&1
            echo "Finished evaluating $model with extraction method $extraction_method and seed $seed"
        done
    done
    # mkdir -p "results_robust_large_fam/no_extraction_sig_seed${seed}"
    # for model in $(ls $models_dir); do
    #     # if model is not directory, skip
    #     if [ ! -d "$models_dir/$model" ]; then
    #         echo "Skipping $model as it is not a directory"
    #         continue
    #     fi
    #     echo "Evaluating $model without extraction selection with si-g and seed $seed"
    #     timeout 3600 prerequisites/venv/bin/python3 robust_pomdps_rl.py --project-path "$models_dir/$model" \
    #         --batched-vec-storm \
    #         --geometric-batched-vec-storm \
    #         --seed $seed \
    #         --without-extraction \
    #         --extraction-method si-g \
    #         > "results_robust_large_fam/no_extraction_sig_seed${seed}/${model}_output.txt" 2>&1
    #     echo "Finished evaluating $model with no extraction method and seed $seed"
    # done
    
done
