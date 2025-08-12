#!/bin/bash

source prerequisites/venv/bin/activate

# This script runs the robust extraction on all models in the specified directory.

folder="models_training_loop"
model_dir="./$folder"

standard_flags="--batched-vec-storm"
restarts=("--periodic-restarts" "")
geometric_flags=("--geometric-batched-vec-storm" "")
extraction_with=("--without-extraction" "")

# Loop through all directories in the model directory
for project_path in "$model_dir"/*/; do
    # Check if the directory is not empty
    if [ -d "$project_path" ] && [ "$(ls -A "$project_path")" ]; then
        # Run the extraction for each model
        echo "Running extraction for model in $project_path"
        
        for extraction_with in "${extraction_with[@]}"; do 
        # if it is without extraction, run only restarts and without restarts. Do not use geometric flags
        if [ "$extraction_with" == "--without-extraction" ]; then
            for restart in "${restarts[@]}"; do
                echo "Running without extraction for model in $project_path with restart: $restart, extraction_with: $extraction_with, and standard flags: $standard_flags"
                timeout 7200 python robust_pomdps_rl.py --project-path "$project_path" $standard_flags $restart $extraction_with
            done
        else
            # Run with geometric flags and restarts
            for geometric_flag in "${geometric_flags[@]}"; do
                for restart in "${restarts[@]}"; do
                    echo "Running extraction for model in $project_path with geometric flag: $geometric_flag and restart: $restart, extraction_with: $extraction_with, and standard flags: $standard_flags"
                    timeout 7200 python robust_pomdps_rl.py --project-path "$project_path" $standard_flags $geometric_flag $restart $extraction_with
                done
            done
        fi

        done
    else
        echo "Skipping empty directory: $project_path"
    fi
done

# Deactivate the virtual environment
deactivate
echo "Extraction completed for all models in $model_dir."
