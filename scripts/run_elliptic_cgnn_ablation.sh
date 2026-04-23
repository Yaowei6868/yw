#!/bin/bash
# run_elliptic_cgnn_ablation.sh — 严格重跑 Elliptic 上的 CGNN backbone 消融
# 用法:
#   bash scripts/run_elliptic_cgnn_ablation.sh
#   nohup bash scripts/run_elliptic_cgnn_ablation.sh > logs/elliptic_cgnn_ablation.log 2>&1 &
#
# 说明:
# - 该脚本不会跳过已存在结果，默认强制重跑
# - 目的是获得 strict controlled ablation:
#     full   = SSF + SPC + SCD
#     noSSF  = SPC + SCD
#     noSPC  = SSF + SCD
#     noSCD  = SSF + SPC

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PYTHON="${PYTHON:-python}"

mkdir -p "${LOG_DIR}"
RUN_COUNT=0
FAIL_COUNT=0
TOTAL_START=$(date +%s)

_yaml_field() { grep -m1 "^${2}:" "$1" | awk '{print $2}' | tr -d '"' | tr -d "'"; }

run_experiment() {
    local config_path="$1"
    local label="$2"
    local exp_name log_file t_start t_end elapsed
    exp_name=$(_yaml_field "$config_path" "name")
    log_file="${LOG_DIR}/${exp_name}_strict_ablation.log"

    echo ""
    echo "  ┌──────────────────────────────────────────────────────┐"
    printf  "  │ %s %-50s│\n" "${label}" "${exp_name}"
    echo "  └──────────────────────────────────────────────────────┘"
    echo "  [RUN ] ${config_path}"
    echo "  [LOG ] logs/$(basename "${log_file}")"

    t_start=$(date +%s)
    cd "${ROOT_DIR}"
    if ${PYTHON} train.py --config "${config_path}" > "${log_file}" 2>&1; then
        t_end=$(date +%s); elapsed=$(( t_end - t_start ))
        printf "  [DONE] %-44s %dm%02ds\n" "${exp_name}" $(( elapsed/60 )) $(( elapsed%60 ))
        RUN_COUNT=$((RUN_COUNT+1))
    else
        echo "  [FAIL] ${exp_name}"
        tail -10 "${log_file}" | sed 's/^/    /'
        FAIL_COUNT=$((FAIL_COUNT+1))
    fi
}

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Elliptic CGNN Strict Ablation                          ║"
printf "║  %-56s║\n" "$(date '+%Y-%m-%d %H:%M:%S')"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  full / noSSF / noSPC / noSCD                           ║"
echo "║  fixed setting: SPC hyperparameters aligned to full     ║"
echo "╚══════════════════════════════════════════════════════════╝"

run_experiment "configs/ours/main/elliptic_TASDCL_CGNN.yaml" "[1/4]"
run_experiment "configs/ours/ablation/elliptic_TASDCL_noSSF_CGNN.yaml" "[2/4]"
run_experiment "configs/ours/ablation/elliptic_TASDCL_noSPC_CGNN.yaml" "[3/4]"
run_experiment "configs/ours/ablation/elliptic_TASDCL_noSCD_CGNN.yaml" "[4/4]"

TOTAL_END=$(date +%s); TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
echo ""
echo "══════════════════════════════════════════════════════════"
printf "  完成: %s | 耗时: %dh%dm%ds\n" \
       "$(date '+%H:%M:%S')" \
       $(( TOTAL_ELAPSED/3600 )) $(( (TOTAL_ELAPSED%3600)/60 )) $(( TOTAL_ELAPSED%60 ))
printf "  ✅ 已完成: %d  ❌ 失败: %d\n" "${RUN_COUNT}" "${FAIL_COUNT}"
echo "══════════════════════════════════════════════════════════"

[[ "${FAIL_COUNT}" -gt 0 ]] && exit 1; exit 0
