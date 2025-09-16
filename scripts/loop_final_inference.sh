#!/bin/bash

BASE_DIR="data/"
# SCRIPT_PATH="/nas-ssd2/vaidehi/projects/MAP/MAP/eval_final_inference.py"
SCRIPT_PATH="/nas-ssd2/vaidehi/projects/MAP/MAP/eval_leakage_exec.py"

# Plan keys and corresponding log suffixes
plan_keys=("run_1_benign_plan" "run_2_sensitive_plan")
log_suffixes=("benign" "sensitive")
log_suffixes=("benign_gemini_qwen_base" "sensitive_gemini_qwen_base")
log_suffixes=("benign_base" "sensitive_base")
log_suffixes=("benign_gpt_qwen_base" "sensitive_gpt_qwen_base")
log_suffixes=("benign_gemini_pro_qwen_base" "sensitive_gemini_pro_qwen_base")
log_suffixes=("benign_gpt_qwen_base" "sensitive_gpt_qwen_base")
log_suffixes=("benign_base" "sensitive_base")
log_suffixes=("benign_qwen_gemini_base" "sensitive_qwen_gemini_base")
log_suffixes=("benign_qwen_gpt_base" "sensitive_qwen_gpt_base")

experiment_ids=()
for i in $(seq 5 124); do
    if [ "$i" -ne 69 ]; then
        experiment_ids+=($i)
    fi
done

eval_files=()
for id in "${experiment_ids[@]}"; do
    file="$BASE_DIR/experiment_${id}_def_adv_both.json"
    if [ -f "$file" ]; then
        eval_files+=("$file")
    else
        echo "⚠️ Skipping missing file: $file"
    fi
done

OUTPUT_FILE="final_inference_results_qwen_gpt_base.json"
echo "[" > "$OUTPUT_FILE"

total=$(( ${#eval_files[@]} * ${#plan_keys[@]} ))
current=0
first=true

show_progress() {
    local progress=$1
    local total=$2
    local width=40
    local percent=$(( 100 * progress / total ))
    local filled=$(( width * progress / total ))
    printf "\r[%-*s] %3d%% (%d/%d)" $width $(printf "%${filled}s" | tr ' ' '=') $percent $progress $total
}

for config_path in "${eval_files[@]}"; do
    for i in "${!plan_keys[@]}"; do
        plan_key="${plan_keys[$i]}"
        log_suffix="${log_suffixes[$i]}"
        ((current++))
        show_progress "$current" "$total"

        result=$(python3 "$SCRIPT_PATH" --config "$config_path" --plan_key "$plan_key" --log_suffix "$log_suffix")
        if [ "$first" = true ]; then
            echo "$result" >> "$OUTPUT_FILE"
            first=false
        else
            echo ",$result" >> "$OUTPUT_FILE"
        fi
    done
done

echo "]" >> "$OUTPUT_FILE"
echo -e "\n✅ Final inference evaluations done. Results saved to $OUTPUT_FILE"
