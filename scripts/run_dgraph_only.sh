#!/bin/bash
# run_dgraph_only.sh — 仅运行 DGraphFin 数据集的所有主对比实验
# 用法: nohup bash run_dgraph_only.sh > logs/dgraph_main.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
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
    csv_path="${SCRIPT_DIR}/${save_dir}/metrics/${exp_name}_aggregate_metrics.csv"
    if [[ -f "$csv_path" ]]; then
        local row_count
        row_count=$(wc -l < "$csv_path" 2>/dev/null || echo 0)
        if [[ "$row_count" -ge "$COMPLETE_ROWS" ]]; then
            return 0
        else
            echo "  [WARN] $exp_name CSV 不完整 (${row_count}/${COMPLETE_ROWS} 行)，重新运行..."
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
    printf  "  │ LOG: logs/%-44s│\n" "${exp_name}.log"
    echo "  └──────────────────────────────────────────────────────┘"
    t_start=$(date +%s)
    cd "${SCRIPT_DIR}"
    if ${PYTHON} train.py --config "${config_path}" > "${log_file}" 2>&1; then
        t_end=$(date +%s); elapsed=$(( t_end - t_start ))
        printf "  [DONE] %-44s %dm%02ds\n" "${exp_name}" $(( elapsed/60 )) $(( elapsed%60 ))
        RUN_COUNT=$((RUN_COUNT+1))
    else
        echo "  [FAIL] ${exp_name} — 查看: ${log_file}"
        tail -10 "${log_file}" | sed 's/^/    /'
        FAIL_COUNT=$((FAIL_COUNT+1))
    fi
}

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  DATASET: DGraphFin — 主对比实验                          ║"
echo "║  $(date '+%Y-%m-%d %H:%M:%S')                           ║"
echo "╚══════════════════════════════════════════════════════════╝"

echo ""; echo "  ── [A] 传统静态基线 ─────────────────────────────────"
run_experiment "configs/traditional/GCN/dgraphfin_Naive_GCN.yaml"
run_experiment "configs/traditional/GAT/dgraphfin_Naive_GAT.yaml"

echo ""; echo "  ── [B] CL 经典基线 ───────────────────────────────────"
run_experiment "configs/traditional/GCN/dgraphfin_EWC_GCN.yaml"
run_experiment "configs/traditional/GCN/dgraphfin_LwF_GCN.yaml"
run_experiment "configs/traditional/GCN/dgraphfin_ER_GCN.yaml"

echo ""; echo "  ── [C] 不平衡处理变体 ────────────────────────────────"
run_experiment "configs/traditional/GraphSMOTE/dgraphfin_Naive_GraphSMOTE.yaml"

echo ""; echo "  ── [D] 核心 SOTA 欺诈检测模型 ───────────────────────"
run_experiment "configs/HOGRL/dgraphfin_Naive_HOGRL.yaml"
run_experiment "configs/HOGRL/dgraphfin_CL_HOGRL.yaml"
run_experiment "configs/CGNN/dgraphfin_Naive_CGNN.yaml"
run_experiment "configs/CGNN/dgraphfin_CL_CGNN.yaml"
run_experiment "configs/ConsisGAD/dgraphfin_Naive_ConsisGAD.yaml"
run_experiment "configs/ConsisGAD/dgraphfin_CL_ConsisGAD.yaml"
run_experiment "configs/Grad/dgraphfin_Naive_Grad.yaml"
run_experiment "configs/Grad/dgraphfin_CL_Grad.yaml"
run_experiment "configs/PMP/dgraphfin_Naive_PMP.yaml"
run_experiment "configs/PMP/dgraphfin_CL_PMP.yaml"
run_experiment "configs/BSL/dgraphfin_Naive_BSL.yaml"

echo ""; echo "  ── [E] BSL CL 策略对比 ───────────────────────────────"
run_experiment "configs/BSL/dgraphfin_EWC_BSL.yaml"
run_experiment "configs/BSL/dgraphfin_LwF_BSL.yaml"
run_experiment "configs/BSL/dgraphfin_ER_BSL.yaml"

echo ""; echo "  ── [F] TASD-CL (Ours) + 消融 ────────────────────────"
run_experiment "configs/BSL/dgraphfin_TASDCL_noSSF_BSL.yaml"
run_experiment "configs/BSL/dgraphfin_TASDCL_noSPC_BSL.yaml"
run_experiment "configs/BSL/dgraphfin_TASDCL_noSCD_BSL.yaml"
run_experiment "configs/BSL/dgraphfin_TASDCL_BSL.yaml"

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
