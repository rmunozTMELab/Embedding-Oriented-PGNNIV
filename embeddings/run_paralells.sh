#!/bin/bash

# Base directory
cd ~/Escritorio/rmunozTMELab/Physically-Guided-Machine-Learning/embeddings

# Relative path for each script
scripts=(
    "autoencoder/main_iterative_ae.py"
    "baseline/main_iterative_baseline.py"
    "fourier/main_iterative_fourier.py"
    "POD/main_iterative_pod.py"
)

# Create an array to store PIDs
pids=()

# Function to launch a script with nohup
run_script() {
    local script_path="$1"
    local dir_path=$(dirname "$script_path")
    local script_file=$(basename "$script_path")
    
    echo "Running $script_file in $dir_path..."
    
    cd "$dir_path"
    nohup python "$script_file" > "${script_file%.py}_output.log" 2>&1 &
    local pid=$!
    echo "PID: $pid"
    pids+=($pid)
    cd - > /dev/null
}

# Run the first 3 scripts
for i in {0..2}; do
    run_script "${scripts[$i]}"
done

# Wait until one of the first 3 scripts finishes
while true; do
    for i in "${!pids[@]}"; do
        if ! kill -0 "${pids[$i]}" 2> /dev/null; then
            echo "Process with PID ${pids[$i]} has finished. Launching the fourth script..."
            unset 'pids[i]'  # Remove the finished PID
            run_script "${scripts[3]}"
            wait "${pids[@]}"  # Wait for the remaining processes
            exit 0
        fi
    done
    sleep 5  # Wait time between checks
done
