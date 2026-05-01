#!/bin/bash
# run_elliptic_actor_final.sh — Elliptic++ Actor 正式实验（全量）
# 用法: nohup bash scripts/run_elliptic_actor_final.sh > logs/elliptic_actor_final.log 2>&1 &
#
# 包含：
#   - PMP Naive（唯一缺失的 baseline）
#   - TASD-CL（主方法，mf10 最优参数）
#   - CL 基线：EWC / LwF / ER on CGNN
#   - 消融：noSSF / noSPC / noSCD
#
# 已有结果（调参确认，跳过）：
#   - GCN/BSL/CGNN/GradGNN/HOGRL/ConsisGAD Naive → 已有
#
# 共 8 个实验

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PYTHON="${PYTHON:-python}"
NUM_TASKS=10
COMPLETE_ROWS=$((NUM_TASKS + 1))

mkdir -p "${LOG_DIR}"
SKIP_COUNT=0; RUN_COUNT=0; FAIL_COUNT=0
TOTAL_START=$(date +%s)

_yaml_field() { grep -m1 "^${2}:" "$1" | awk '{print $2}' | tr -d '"' | tr -d "'"; }

should_skip() {
    local config_path="$1"
    local exp_name save_dir csv_path
    exp_name=$(_yaml_field "$config_path" "name")
    save_dir=$(grep -m1 "save_dir:" "$config_path" | awk '{print $2}' | tr -d '"' | tr -d "'")
    csv_path="${ROOT_DIR}/${save_dir}/metrics/${exp_name}_aggregate_metrics.csv"
    if [[ -f "$csv_path" ]]; then
        local row_count
        row_count=$(wc -l < "$csv_path" 2>/dev/null || echo 0)
        if [[ "$row_count" -ge "$COMPLETE_ROWS" ]]; then
            return 0
        fi
    fi
    return 1
}

run_experiment() {
    local config_path="$1"
    local exp_name log_file t_start t_end elapsed
    exp_name=$(_yaml_field "$config_path" "name")
    log_file="${LOG_DIR}/${exp_name}.log"

    if should_skip "$config_path"; then
        echo "  [SKIP] ${exp_name}"; SKIP_COUNT=$((SKIP_COUNT+1)); return
    fi

    echo ""
    echo "  ┌──────────────────────────────────────────────────────┐"
    printf  "  │ RUN: %-49s│\n" "${exp_name}"
    echo "  └──────────────────────────────────────────────────────┘"
    t_start=$(date +%s)
    cd "${ROOT_DIR}"
    if ${PYTHON} train.py --config "${config_path}" > "${log_file}" 2>&1; then
        t_end=$(date +%s); elapsed=$(( t_end - t_start ))
        printf "  [DONE] %-44s %dm%02ds\n" "${exp_name}" $(( elapsed/60 )) $(( elapsed%60 ))
        RUN_COUNT=$((RUN_COUNT+1))
    else
        echo "  [FAIL] ${exp_name}"
        tail -5 "${log_file}" | sed 's/^/    /'
        FAIL_COUNT=$((FAIL_COUNT+1))
    fi
}

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Elliptic++ Actor 正式实验（全量）                       ║"
printf "║  %-56s║\n" "$(date '+%Y-%m-%d %H:%M:%S')"
echo "╚══════════════════════════════════════════════════════════╝"

# ── 缺失的 Baseline ──────────────────────────────────────────
echo ""; echo "  ── Baseline: PMP ───────────────────────────────────"
run_experiment "configs/fraud_sota/elliptic_actor/elliptic_actor_Naive_PMP.yaml"

# ── 主方法：TASD-CL ──────────────────────────────────────────
echo ""; echo "  ── 主方法：TASD-CL (CGNN) ──────────────────────────"
run_experiment "configs/ours/main/elliptic_actor_TASDCL_CGNN.yaml"

# ── CL 基线（CGNN backbone）─────────────────────────────────
echo ""; echo "  ── CL 基线：EWC / LwF / ER ─────────────────────────"
run_experiment "configs/ours/cl_on_cgnn/elliptic_actor_EWC_CGNN.yaml"
run_experiment "configs/ours/cl_on_cgnn/elliptic_actor_LwF_CGNN.yaml"
run_experiment "configs/ours/cl_on_cgnn/elliptic_actor_ER_CGNN.yaml"

# ── 消融实验 ─────────────────────────────────────────────────
echo ""; echo "  ── 消融实验：noSSF / noSPC / noSCD ─────────────────"
run_experiment "configs/ours/ablation/elliptic_actor_TASDCL_noSSF_CGNN.yaml"
run_experiment "configs/ours/ablation/elliptic_actor_TASDCL_noSPC_CGNN.yaml"
run_experiment "configs/ours/ablation/elliptic_actor_TASDCL_noSCD_CGNN.yaml"

TOTAL_END=$(date +%s); TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
echo ""
echo "══════════════════════════════════════════════════════════"
printf "  完成: %s | 耗时: %dh%dm%ds\n" \
       "$(date '+%H:%M:%S')" \
       $(( TOTAL_ELAPSED/3600 )) $(( (TOTAL_ELAPSED%3600)/60 )) $(( TOTAL_ELAPSED%60 ))
printf "  ✅ 已完成: %d  ⏭ 已跳过: %d  ❌ 失败: %d\n" \
       "${RUN_COUNT}" "${SKIP_COUNT}" "${FAIL_COUNT}"
echo "══════════════════════════════════════════════════════════"

[[ "${FAIL_COUNT}" -gt 0 ]] && exit 1; exit 0
