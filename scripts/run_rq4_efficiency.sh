#!/bin/bash
# RQ4 efficiency experiments. The Trainer writes time, GPU memory, model size,
# stored memory, and inference-time columns into each aggregate CSV.
#
# This script runs two efficiency groups:
#   1) system_level: cumulative/full-graph baselines vs task-only TASD-CL.
#   2) task_only_overhead: all methods under the same task-only snapshots.
#
# Usage:
#   bash scripts/run_rq4_efficiency.sh
#   FORCE_RERUN=1 nohup bash scripts/run_rq4_efficiency.sh > logs/rq4_efficiency.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PYTHON="${PYTHON:-python}"

mkdir -p "${LOG_DIR}"
SKIP_COUNT=0
RUN_COUNT=0
FAIL_COUNT=0
TOTAL_START=$(date +%s)

yaml_field() {
    grep -m1 "^${2}:" "$1" | awk '{print $2}' | tr -d '"' | tr -d "'"
}

yaml_save_dir() {
    grep -m1 "save_dir:" "$1" | awk '{print $2}' | tr -d '"' | tr -d "'"
}

run_efficiency() {
    local config_path="$1"
    local exp_name save_dir
    exp_name=$(yaml_field "$config_path" "name")
    save_dir=$(yaml_save_dir "$config_path")
    local csv_path="${ROOT_DIR}/${save_dir}/metrics/${exp_name}_aggregate_metrics.csv"
    local log_file="${LOG_DIR}/${exp_name}.log"
    local row_count t_start t_end elapsed

    if [[ "${FORCE_RERUN:-0}" != "1" && -f "$csv_path" ]]; then
        row_count=$(wc -l < "$csv_path" 2>/dev/null || echo 0)
        if [[ "$row_count" -ge 11 ]]; then
            echo "  [SKIP] ${exp_name}"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            return
        fi
    fi

    echo ""
    echo "  [RUN ] ${exp_name}"
    echo "  [BASE] ${config_path}"
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

echo "============================================================"
echo " RQ4 efficiency experiments"
printf " %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

echo ""
echo "------------------------------------------------------------"
echo " Table A: system-level deployment protocols"
echo "------------------------------------------------------------"
for config in "${ROOT_DIR}"/configs/elliptic/efficiency/system_level/*.yaml; do
    run_efficiency "${config#${ROOT_DIR}/}"
done

echo ""
echo "------------------------------------------------------------"
echo " Table B: task-only model overhead"
echo "------------------------------------------------------------"
for config in "${ROOT_DIR}"/configs/elliptic/efficiency/task_only_overhead/*.yaml; do
    run_efficiency "${config#${ROOT_DIR}/}"
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
echo ""
echo "============================================================"
printf " Finished: %s | Time: %dh%dm%ds\n" \
       "$(date '+%H:%M:%S')" \
       $((TOTAL_ELAPSED / 3600)) $(((TOTAL_ELAPSED % 3600) / 60)) $((TOTAL_ELAPSED % 60))
printf " Done: %d | Skipped: %d | Failed: %d\n" "${RUN_COUNT}" "${SKIP_COUNT}" "${FAIL_COUNT}"
echo "============================================================"

[[ "${FAIL_COUNT}" -gt 0 ]] && exit 1
exit 0
