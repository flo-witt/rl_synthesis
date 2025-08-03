source prerequisites/venv/bin/activate

model_dir=./models_robust_subset
methods=("alergia" "si-g" "si-t" "bottleneck")
batched_vec_storm=(True False)
seeds=10

lstm_widths=(8 32)


create_log_folder(){
    if ! [ -d $1 ]; then
        mkdir $1
    fi
}



run_loop(){
    # Go through all models in the folder and runs paynt on the model
    folders=($1)
    experiment_folder="experiments_robust_extraction"
    create_log_folder $experiment_folder
    for seed in $(seq 1 $seeds); do
        for width in ${lstm_widths[@]}; do
            for method in ${methods[@]}; do
                for batched_vec in ${batched_vec_storm[@]}; do
                    for entry in `ls $model_dir`; do
                        echo Running robust extraction on $model_dir/$entry with method $method, width $width, batched_vec $batched_vec
                        #check, if the entry is folder
                        if ! [ -d $model_dir/$entry ]; then
                            continue
                        fi
                        model_name=$entry
                        model_path=$model_dir/$entry
                        log_file_name="./$experiment_folder/$model_name-$method-$width-$batched_vec-$seed.log"
                        experiment_datafile_name="./$experiment_folder/$model_name-$method-$width-$batched_vec-$seed.data"
                        err_log_file_name="./$experiment_folder/$model_name-$method-$width-$batched_vec-$seed.err.log"
                        if [ $batched_vec ]; then
                            batched_vec_flag="--batched-vec-storm"
                        else
                            batched_vec_flag=""
                        fi
                        echo "Running: python3 robust_pomdps_rl.py --project-path $model_path $batched_vec_flag --extraction-method $method --lstm-width $width"
                        start_time=$(date +%s)
                        python3 robust_pomdps_rl.py --project-path $model_path $batched_vec_flag --extraction-method $method --lstm-width $width > $log_file_name 2> $err_log_file_name
                        end_time=$(date +%s)
                        elapsed_time=$(($end_time - $start_time))
                        echo "Finished running $model_name with method $method, width $width, batched_vec $batched_vec in $elapsed_time seconds"
                        echo "Log file: $log_file_name"
                        echo "Error log file: $err_log_file_name"
                        echo "Experiment data file: $experiment_datafile_name"
                        # Save the experiment data
                        echo "Model: $model_name, Method: $method, Width: $width, Batched Vec: $batched_vec, Seed: $seed, Elapsed Time: $elapsed_time seconds" > $experiment_datafile_name
                    done
                done    
            done
        done
    done

}

# Run the loop with the model directories
run_loop $model_dir
