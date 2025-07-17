#!/bin/bash

# Base directory
cd ~/Escritorio/rmunozTMELab/Physically-Guided-Machine-Learning/embeddings

# Relative path for each script
scripts=(
    "autoencoder/main_iterative_ae.py"      # index 0
    "baseline/main_iterative_baseline.py"   # index 1
    "fourier/main_iterative_fourier.py"     # index 2
    "POD/main_iterative_pod.py"             # index 3
)

# Array to track PIDs and names
declare -A pid_to_script

# Function to launch a script with nohup
run_script() {
    local script_path="$1"
    local dir_path=$(dirname "$script_path")
    local script_file=$(basename "$script_path")
    
    echo "Running $script_file in $dir_path..."
    
    cd "$dir_path"
    nohup python "$script_file" > "${script_file%.py}_output.log" 2>&1 &
    local pid=$!
    pid_to_script[$pid]="$script_path"
    echo "PID: $pid for $script_file"
    cd - > /dev/null
}

# Start baseline and fourier
run_script "${scripts[1]}"  # baseline
pid_baseline=$!

run_script "${scripts[2]}"  # fourier
pid_fourier=$!

# Monitor both and launch next scripts accordingly
autoencoder_launched=false
pod_launched=false

while [[ ${#pid_to_script[@]} -gt 0 ]]; do
    for pid in "${!pid_to_script[@]}"; do
        if ! kill -0 "$pid" 2> /dev/null; then
            finished_script="${pid_to_script[$pid]}"
            echo "Process $finished_script (PID $pid) finished."

            # Handle dependencies
            if [[ "$finished_script" == "${scripts[1]}" && $autoencoder_launched == false ]]; then
                run_script "${scripts[0]}"  # autoencoder
                autoencoder_launched=true
            elif [[ "$finished_script" == "${scripts[2]}" && $pod_launched == false ]]; then
                run_script "${scripts[3]}"  # POD
                pod_launched=true
            fi

            unset pid_to_script[$pid]
        fi
    done
    sleep 5
done

echo "All processes completed."
