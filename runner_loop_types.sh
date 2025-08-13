#!/bin/bash

trap 'echo "Caught Ctrl+C, killing all subprocesses..."; pkill -P $$; exit 1' SIGINT

# This script runs the robust extraction on all models in the specified directory.

folder="models_training_loop"
model_dir="./$folder"

standard_flags="--batched-vec-storm"
restarts=("--periodic-restarts" "")
geometric_flags=("--geometric-batched-vec-storm" "")
extraction_with=("--without-extraction" "")
seeds=10

log_folder="experiments_logs_loop_types"
# Create the log folder if it doesn't exist
mkdir -p "$log_folder"

for i in $(seq 1 $seeds); do
    # Loop through all directories in the model directory
    for project_path in "$model_dir"/*/; do
        # Check if the directory is not empty
        model_name=$(basename "$project_path")
        if [ -d "$project_path" ] && [ "$(ls -A "$project_path")" ]; then
            # Run the extraction for each model
            echo "Running extraction for model in $project_path"
            
            for extraction_with in "${extraction_with[@]}"; do 
                # if it is without extraction, run only restarts and without restarts. Do not use geometric flags
                if [ "$extraction_with" == "--without-extraction" ]; then
                    for restart in "${restarts[@]}"; do
                        echo "Running without extraction for model in $project_path with restart: $restart, extraction_with: $extraction_with, and standard flags: $standard_flags"
                        timeout 7200 prerequisites/venv/bin/python3.10 \
                            robust_pomdps_rl.py \
                            --project-path "$project_path" \
                            $standard_flags \
                            $restart \
                            $extraction_with \
                            > "$log_folder/${model_name}_without_extraction_restart_${restart}_${i}.log" \
                            2> "$log_folder/${model_name}_without_extraction_restart_${restart}_${i}.err"
                    done
                else
                    # Run with geometric flags and restarts
                    for geometric_flag in "${geometric_flags[@]}"; do
                        for restart in "${restarts[@]}"; do
                            echo "Running extraction for model in $project_path with geometric flag: $geometric_flag and restart: $restart, extraction_with: $extraction_with, and standard flags: $standard_flags"
                            timeout 7200 prerequisites/venv/bin/python3.10 \
                                robust_pomdps_rl.py \
                                --project-path "$project_path" \
                                $standard_flags \
                                $geometric_flag \
                                $restart \
                                $extraction_with \
                                > "$log_folder/${model_name}_geometric_${geometric_flag}_restart_${restart}_${i}.log" \
                                2> "$log_folder/${model_name}_geometric_${geometric_flag}_restart_${restart}_${i}.err"
                        done
                    done
                fi
            done
        else
            echo "Skipping empty directory: $project_path"
        fi
    done
done

# Deactivate the virtual environment
echo "Extraction completed for all models in $model_dir."
