#!/bin/bash
# run_elliptic_cl_baselines.sh
# Generic continual learning baselines on ordinary GCN for Elliptic.
#
# Usage:
#   bash scripts/run_elliptic_cl_baselines.sh
#   nohup bash scripts/run_elliptic_cl_baselines.sh > logs/elliptic_cl_baselines.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PYTHON="${PYTHON:-python}"
NUM_TASKS=10
COMPLETE_ROWS=$((NUM_TASKS + 1))

mkdir -p "${LOG_DIR}"
SKIP_COUNT=0
RUN_COUNT=0
FAIL_COUNT=0
TOTAL_START=$(date +%s)

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
        SKIP_COUNT=$((SKIP_COUNT + 1))
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
        printf "  [DONE] %-32s %dm%02ds\n" "${exp_name}" $((elapsed / 60)) $((elapsed % 60))
        RUN_COUNT=$((RUN_COUNT + 1))
    else
        echo "  [FAIL] ${exp_name}"
        tail -10 "${log_file}" | sed 's/^/    /'
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

echo ""
echo "============================================================"
echo " Elliptic generic CL baselines on ordinary GCN"
echo " EWC / LwF / ER"
printf " %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

run_experiment "configs/ours/cl_on_gcn/elliptic_EWC_GCN.yaml"
run_experiment "configs/ours/cl_on_gcn/elliptic_LwF_GCN.yaml"
run_experiment "configs/ours/cl_on_gcn/elliptic_ER_GCN.yaml"

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "============================================================"
printf " Finished: %s | Time: %dh%dm%ds\n" \
       "$(date '+%H:%M:%S')" \
       $((TOTAL_ELAPSED / 3600)) $(((TOTAL_ELAPSED % 3600) / 60)) $((TOTAL_ELAPSED % 60))
printf " Done: %d | Skipped: %d | Failed: %d\n" \
       "${RUN_COUNT}" "${SKIP_COUNT}" "${FAIL_COUNT}"
echo "============================================================"

[[ "${FAIL_COUNT}" -gt 0 ]] && exit 1
exit 0
