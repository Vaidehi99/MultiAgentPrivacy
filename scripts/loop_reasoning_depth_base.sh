#!/bin/bash

BASE_DIR="/nas-ssd2/vaidehi/projects/MAP/peoplejoin/workspace/peoplejoin-qa/experiments/adversarial_scenarios"
SCRIPT_PATH="/nas-ssd2/vaidehi/projects/MAP/MAP/reasoning_depth.py"

# Plan keys and corresponding log suffixes
log_suffixes=("benign_qwen_gemini_base" "sensitive_qwen_gemini_base")

# List of experiment IDs to process (skip 69)
experiment_ids=()
for i in $(seq 5 124); do
    if [ "$i" -ne 69 ]; then
        experiment_ids+=($i)
    fi
done

# Collect all existing evaluation files
eval_files=()
for id in "${experiment_ids[@]}"; do
    file="$BASE_DIR/experiment_${id}_def_adv_both.json"
    if [ -f "$file" ]; then
        eval_files+=("$file")
    else
        echo "Skipping missing file: $file"
    fi
done

OUTPUT_FILE="reasoning_depth_binary_results_qwen_gemini_base.json"
echo "[" > "$OUTPUT_FILE"

total=$(( ${#eval_files[@]} * ${#log_suffixes[@]} ))
current=0
first=true

# Progress bar function
show_progress() {
    local progress=$1
    local total=$2
    local width=40
    local percent=$(( 100 * progress / total ))
    local filled=$(( width * progress / total ))
    printf "\r[%-*s] %3d%% (%d/%d)" $width $(printf "%${filled}s" | tr ' ' '=') $percent $progress $total
}

# Run evaluations
for config_path in "${eval_files[@]}"; do
    for i in "${!log_suffixes[@]}"; do
        log_suffix="${log_suffixes[$i]}"
        ((current++))
        show_progress "$current" "$total"

        result=$(python3 "$SCRIPT_PATH" --config "$config_path" --log_suffix "$log_suffix")
        if [ "$first" = true ]; then
            echo "$result" >> "$OUTPUT_FILE"
            first=false
        else
            echo ",$result" >> "$OUTPUT_FILE"
        fi
    done
done

echo "]" >> "$OUTPUT_FILE"
echo -e "\n✅ Reasoning depth evaluations done. Results saved to $OUTPUT_FILE"
