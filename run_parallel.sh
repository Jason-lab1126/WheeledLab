#!/bin/bash

SCRIPT="/home/urviru/WheeledLab/source/wheeledlab_rl/scripts/train_rl.py"
DATA_DIR="/home/yandabao/Desktop/WheeledLab-research/source/wheeledlab_tasks/wheeledlab_tasks/navigation/config/agents/mushr/yanda_data/expert_obs_masked_8_demos.pt"

seed=(40 50 60)
entropy_vals=(0.03 0.04 0.05)

MAX_IDLE=600
MAX_JOBS=3
CHECK_INTERVAL=10
LAUNCH_DELAY=5

mkdir -p auto
JOB_TRACKER="auto/running_jobs_$$.txt"
rm -f "$JOB_TRACKER"

count_active_jobs() {
    local count=0
    if [[ -f "$JOB_TRACKER" ]]; then
        while IFS='|' read -r pid _; do
            if kill -0 "$pid" 2>/dev/null; then
                ((count++))
            fi
        done < "$JOB_TRACKER"
    fi
    echo "$count"
}

for s in "${seed[@]}"; do
    for ent in "${entropy_vals[@]}"; do
        while (( $(count_active_jobs) >= MAX_JOBS )); do
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] At max capacity ($MAX_JOBS jobs). Waiting..."
            sleep 10
        done

        log_file="auto/log_entropy${ent}_seed${s}_$(date +%s).txt"
        full_cmd="python ${SCRIPT} --headless -r RSS_DRIFT_CONFIG agent.seed=${s} agent.algorithm.entropy_coef=${ent}"
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: $full_cmd"
        echo "  Logging to: $log_file"
        
        bash -c "$full_cmd" > "$log_file" 2>&1 &
        pid=$!
        
        echo "${pid}|${log_file}|${full_cmd}" >> "$JOB_TRACKER"
        
        (
            while kill -0 $pid 2>/dev/null; do
                sleep $CHECK_INTERVAL
                
                if [[ -f "$log_file" ]]; then
                    last_mod=$(stat -c %Y "$log_file" 2>/dev/null || echo 0)
                    now=$(date +%s)
                    idle_time=$((now - last_mod))
                    
                    if (( idle_time > MAX_IDLE )); then
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] No output for ${idle_time}s. Killing PID $pid (seed=$s, entropy=$ent)"
                        kill -9 $pid 2>/dev/null
                        break
                    fi
                fi
            done
            
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Command done for seed=$s, entropy_coef=$ent"
        ) &
        
        sleep $LAUNCH_DELAY
    done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All jobs launched. Waiting for completion..."
wait

rm -f "$JOB_TRACKER"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All runs complete."
