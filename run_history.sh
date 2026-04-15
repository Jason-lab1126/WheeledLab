#!/bin/bash

SCRIPT="/home/urviru/WheeledLab/source/wheeledlab_rl/scripts/train_rl.py"
DATA_DIR="/home/yandabao/Desktop/WheeledLab-research/source/wheeledlab_tasks/wheeledlab_tasks/navigation/config/agents/mushr/yanda_data/expert_obs_masked_8_demos.pt"
seed=(40 50 60)

MAX_IDLE=600  # seconds of no output before considering the run frozen

for s in "${seed[@]}"; do
    log_file="auto/log_$(date +%s).txt"
    full_cmd="python ${SCRIPT} --headless -r RSS_RECURRENT_DRIFT_CONFIG agent.seed=${s}"

    # Run command in background and redirect output
    echo "Running: $full_cmd"
    echo "Logging to: $log_file"
    bash -c "$full_cmd" > "$log_file" 2>&1 &
    pid=$!

    last_mod_time=$(stat -c %Y "$log_file")

    while kill -0 $pid 2>/dev/null; do
        sleep 10
        new_mod_time=$(stat -c %Y "$log_file")
        now=$(date +%s)

        if (( now - new_mod_time > MAX_IDLE )); then
            echo "No output for $MAX_IDLE seconds. Killing process: $cmd"
            kill -9 $pid
            break
        fi

        last_mod_time=$new_mod_time
    done

    echo "Command done: $cmd"
done
