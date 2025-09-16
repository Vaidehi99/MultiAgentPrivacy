#!/bin/bash

# Construct experiment_ids from 5 to 125 excluding 69
experiment_ids=()
for i in $(seq 5 124); do
    if [ "$i" -ne 69 ]; then
        experiment_ids+=($i)
    fi
done

goals=(benign)  # example goals

base_config_path="data/"
total_jobs=$((${#experiment_ids[@]} * ${#goals[@]}))
current=0

# Function to show progress bar
show_progress() {
    local progress=$1
    local total=$2
    local width=40
    local percent=$(( 100 * progress / total ))
    local filled=$(( width * progress / total ))
    local empty=$(( width - filled ))
    printf "\r[%-*s%s] %3d%% (%d/%d)" $width $(printf "%${filled}s" | tr ' ' '=') ">" $percent $progress $total
}

# Loop with progress
for id in "${experiment_ids[@]}"; do
    config_file="${base_config_path}/experiment_${id}_def_adv_both.json"

    for goal in "${goals[@]}"; do
        ((current++))
        show_progress "$current" "$total_jobs"
        python3 scripts/qwen_gemini_map_codef.py --config "${config_file}" --goal "${goal}"
    done
done

echo -e "\n✅ All experiments completed!"
