#!/bin/bash
# run_actor_tasd_opt.sh
# TASD-CL Actor optimization sweep for AUC-ROC, G-Mean, and MacroF1.
#
# Usage:
#   nohup bash scripts/run_actor_tasd_opt.sh > logs/actor_tasd_opt.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PYTHON="${PYTHON:-python}"
NUM_TASKS=10
COMPLETE_ROWS=$((NUM_TASKS + 1))

mkdir -p "${LOG_DIR}"

CONFIGS=(
  "configs/ours/actor_opt/actor_TASDCL_spc_mf10.yaml"
  "configs/ours/actor_opt/actor_TASDCL_spc_mf10_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_spc_mf10_th40.yaml"
  "configs/ours/actor_opt/actor_TASDCL_spc_mf10_th35.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc3_th50.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc3_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc3_th40.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc3_th35.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc3_scd02_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc3_tau03_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_th50.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_th40.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_th35.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_scd02_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_scd02_th40.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_scd03_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc5_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc5_th40.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_th42.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_th43.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_th44.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_th46.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_th47.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc5_th43.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc5_th44.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc5_th46.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_scd04_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc45_scd05_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc45_scd04_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc45_scd06_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_scd06_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc5_scd04_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc5_scd02_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_lr8_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_drop04_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc4_h256_th45.yaml"
  "configs/ours/actor_opt/actor_TASDCL_nospc_ewc3_lr5_th45.yaml"
)

yaml_field() {
    grep -m1 "^${2}:" "$1" | awk '{print $2}' | tr -d '"' | tr -d "'"
}

should_skip() {
    local config_path="$1"
    local exp_name save_dir csv_path row_count

    if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
        return 1
    fi

    exp_name=$(yaml_field "$config_path" "name")
    save_dir=$(grep -m1 "save_dir:" "$config_path" | awk '{print $2}' | tr -d '"' | tr -d "'")
    csv_path="${ROOT_DIR}/${save_dir}/metrics/${exp_name}_aggregate_metrics.csv"

    if [[ -f "$csv_path" ]]; then
        row_count=$(wc -l < "$csv_path" 2>/dev/null || echo 0)
        if [[ "$row_count" -ge "$COMPLETE_ROWS" ]]; then
            return 0
        fi
        echo "  [WARN] ${exp_name} has incomplete CSV (${row_count}/${COMPLETE_ROWS} rows); rerunning."
    fi
    return 1
}

run_experiment() {
    local config_path="$1"
    local exp_name log_file t_start t_end elapsed
    exp_name=$(yaml_field "$config_path" "name")
    log_file="${LOG_DIR}/${exp_name}.log"

    if should_skip "$config_path"; then
        echo "  [SKIP] ${exp_name}"
        return
    fi

    echo ""
    echo "  [RUN ] ${exp_name}"
    echo "  [CONF] ${config_path}"
    echo "  [LOG ] logs/$(basename "${log_file}")"

    t_start=$(date +%s)
    cd "${ROOT_DIR}"
    if ${PYTHON} train.py --config "${config_path}" > "${log_file}" 2>&1; then
        t_end=$(date +%s)
        elapsed=$((t_end - t_start))
        printf "  [DONE] %-42s %dm%02ds\n" "${exp_name}" $((elapsed / 60)) $((elapsed % 60))
    else
        echo "  [FAIL] ${exp_name}"
        tail -10 "${log_file}" | sed 's/^/    /'
        return 1
    fi
}

echo ""
echo "============================================================"
echo " TASD-CL Actor optimization sweep"
printf " %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
echo " Target metrics: AUC-ROC / G-Mean / MacroF1"
echo "============================================================"

FAIL_COUNT=0
for config_path in "${CONFIGS[@]}"; do
    run_experiment "$config_path" || FAIL_COUNT=$((FAIL_COUNT + 1))
done

echo ""
echo "============================================================"
echo " Sweep finished. Failed: ${FAIL_COUNT}"
echo "============================================================"

exit "${FAIL_COUNT}"
