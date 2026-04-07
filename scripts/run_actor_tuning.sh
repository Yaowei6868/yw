#!/bin/bash
# run_actor_tuning.sh — Elliptic++ Actor 全面调参实验
# 用法: nohup bash scripts/run_actor_tuning.sh > logs/actor_tuning.log 2>&1 &
#
# 阶段说明：
#   Stage 1: 修复 CGNN 收敛（7 个变体）
#   Stage 2: 其他 baseline 调参（GCN×3, BSL×2, Grad×2, HOGRL×2）
#   Stage 3: TASD-CL 调参（基于最优 CGNN 参数，3 个变体）
# 共 19 个实验

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
echo "║  Elliptic++ Actor 全面调参                               ║"
printf "║  %-56s║\n" "$(date '+%Y-%m-%d %H:%M:%S')"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Stage 1: CGNN 收敛修复 ─────────────────────────────────────
echo ""; echo "  ── Stage 1: CGNN 收敛调参 (7个) ──────────────────"
run_experiment "configs/tuning/elliptic++actor/cgnn/cgnn_noaux.yaml"
run_experiment "configs/tuning/elliptic++actor/cgnn/cgnn_e50.yaml"
run_experiment "configs/tuning/elliptic++actor/cgnn/cgnn_e100.yaml"
run_experiment "configs/tuning/elliptic++actor/cgnn/cgnn_e50_noaux.yaml"
run_experiment "configs/tuning/elliptic++actor/cgnn/cgnn_e50_lr5.yaml"
run_experiment "configs/tuning/elliptic++actor/cgnn/cgnn_e50_lr1.yaml"
run_experiment "configs/tuning/elliptic++actor/cgnn/cgnn_e100_noaux_lr5.yaml"

# ── Stage 2: 其他 baseline 调参 ────────────────────────────────
echo ""; echo "  ── Stage 2: 其他 Baseline 调参 (9个) ──────────────"
run_experiment "configs/tuning/elliptic++actor/gcn/gcn_e50.yaml"
run_experiment "configs/tuning/elliptic++actor/gcn/gcn_e100.yaml"
run_experiment "configs/tuning/elliptic++actor/gcn/gcn_e50_h256.yaml"
run_experiment "configs/tuning/elliptic++actor/bsl/bsl_e50.yaml"
run_experiment "configs/tuning/elliptic++actor/bsl/bsl_e50_h128.yaml"
run_experiment "configs/tuning/elliptic++actor/grad/grad_e50.yaml"
run_experiment "configs/tuning/elliptic++actor/grad/grad_e50_gs16.yaml"
run_experiment "configs/tuning/elliptic++actor/hogrl/hogrl_e50.yaml"
run_experiment "configs/tuning/elliptic++actor/hogrl/hogrl_e50_lr3.yaml"

# ── Stage 3: TASD-CL 调参 ──────────────────────────────────────
echo ""; echo "  ── Stage 3: TASD-CL 调参 (3个) ─────────────────"
run_experiment "configs/tuning/elliptic++actor/tasdcl/tasdcl_noaux_e50.yaml"
run_experiment "configs/tuning/elliptic++actor/tasdcl/tasdcl_noaux_e50_lam05.yaml"
run_experiment "configs/tuning/elliptic++actor/tasdcl/tasdcl_noaux_e50_nospc.yaml"

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
