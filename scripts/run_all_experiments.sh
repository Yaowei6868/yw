#!/bin/bash
# =============================================================================
# run_all_experiments.sh
# TASD-CL 主对比实验自动化脚本
#
# 使用方法:
#   # 方式1: 前台运行 (需保持 SSH 连接)
#   bash run_all_experiments.sh
#
#   # 方式2: nohup 后台运行 (推荐 — 断开 SSH 后继续)
#   nohup bash run_all_experiments.sh > logs/main.log 2>&1 &
#   echo "PID: $!"          # 记录进程号，用于 kill
#
#   # 方式3: tmux (推荐用于调试)
#   tmux new -s tasdcl
#   bash run_all_experiments.sh
#   # Ctrl+B D 分离; tmux attach -t tasdcl 重新连接
#
# 断点续跑: 脚本在每个实验前检查结果 CSV 是否已完整，
#            若完整 (含10行任务数据) 则自动跳过。
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# 基础配置
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PYTHON="${PYTHON:-python}"          # 可用 PYTHON=python3 bash run_all_experiments.sh 覆盖
NUM_TASKS=10                        # Elliptic / Elliptic++Actor 均为 10 个任务
COMPLETE_ROWS=$((NUM_TASKS + 1))    # CSV 行数: 1 表头 + 10 数据行

mkdir -p "${LOG_DIR}"

# 记录总开始时间
TOTAL_START=$(date +%s)
SKIP_COUNT=0
RUN_COUNT=0
FAIL_COUNT=0

# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------

# 从 yaml 安全地提取字段值
_yaml_field() {
    local file="$1" field="$2"
    grep -m1 "^${field}:" "$file" | awk '{print $2}' | tr -d '"' | tr -d "'"
}

# 核心: 判断实验是否已完成 (断点续跑逻辑)
# 返回 0 = 跳过, 1 = 需要运行
should_skip() {
    local config_path="$1"
    local exp_name
    local save_dir
    exp_name=$(_yaml_field "$config_path" "name")
    save_dir=$(grep -m1 "save_dir:" "$config_path" | awk '{print $2}' | tr -d '"' | tr -d "'")

    # 路径处理: save_dir 可能是相对路径
    local csv_path="${SCRIPT_DIR}/${save_dir}/metrics/${exp_name}_aggregate_metrics.csv"

    if [[ -f "$csv_path" ]]; then
        local row_count
        row_count=$(wc -l < "$csv_path" 2>/dev/null || echo 0)
        if [[ "$row_count" -ge "$COMPLETE_ROWS" ]]; then
            return 0   # 跳过
        else
            echo "  [WARN] $exp_name CSV 不完整 (${row_count}/${COMPLETE_ROWS} 行)，重新运行..."
        fi
    fi
    return 1  # 需要运行
}

# 运行单个实验
run_experiment() {
    local config_path="$1"
    local exp_name
    exp_name=$(_yaml_field "$config_path" "name")
    local log_file="${LOG_DIR}/${exp_name}.log"

    if should_skip "$config_path"; then
        echo "  [SKIP] ${exp_name}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return
    fi

    echo ""
    echo "  ┌─────────────────────────────────────────────────────┐"
    printf  "  │ %-53s│\n" "RUN: ${exp_name}"
    printf  "  │ %-53s│\n" "LOG: logs/${exp_name}.log"
    echo "  └─────────────────────────────────────────────────────┘"

    local t_start
    t_start=$(date +%s)

    # 运行实验，stdout+stderr 均写入独立日志文件
    cd "${SCRIPT_DIR}"
    if ${PYTHON} train.py --config "${config_path}" > "${log_file}" 2>&1; then
        local t_end elapsed
        t_end=$(date +%s)
        elapsed=$(( t_end - t_start ))
        printf "  [DONE] %-45s %dm%02ds\n" \
               "${exp_name}" $(( elapsed/60 )) $(( elapsed%60 ))
        RUN_COUNT=$((RUN_COUNT + 1))
    else
        local exit_code=$?
        echo "  [FAIL] ${exp_name} — exit code ${exit_code}"
        echo "         详细日志: ${log_file}"
        echo "         最后10行:"
        tail -10 "${log_file}" | sed 's/^/         /'
        FAIL_COUNT=$((FAIL_COUNT + 1))
        # 不 exit，继续跑后续实验
    fi
}

# 打印分组标题
section() {
    echo ""
    echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
echo ""
echo "██████████████████████████████████████████████████████████"
echo "█  TASD-CL 主对比实验 — 自动化运行脚本                     █"
echo "█  启动时间: $(date '+%Y-%m-%d %H:%M:%S')                  █"
echo "█  Python:  $(${PYTHON} --version 2>&1)                    █"
echo "████████████████████████████████████████████████████████ ██"

# ======================================================================
# ★ PART 1: Elliptic Dataset
# ======================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  DATASET 1/3 — Elliptic                                  ║"
echo "╚══════════════════════════════════════════════════════════╝"

section "[A] 传统静态基线 (Sequential Fine-Tuning)"
run_experiment "configs/traditional/elliptic_Naive_GCN.yaml"
run_experiment "configs/traditional/elliptic_Naive_GAT.yaml"

section "[B] 持续学习经典基线 (CL Baselines)"
run_experiment "configs/cl_baselines/elliptic_EWC_GCN.yaml"
run_experiment "configs/cl_baselines/elliptic_LwF_GCN.yaml"
run_experiment "configs/cl_baselines/elliptic_CL_GCN.yaml"        # ER-GNN (AAAI'21)

section "[C] 不平衡处理变体 (Imbalance Handling)"
# 注: GraphSMOTE 此处使用 GCN+GCNConv 骨架 + 动态 Focal Loss
# 与原版 SMOTE 过采样有区别，详见 models.py GraphSMOTE 类说明
run_experiment "configs/imbalanced/elliptic_Naive_GraphSMOTE.yaml"

section "[D] 核心 SOTA 欺诈检测模型 (2024-2025)"
run_experiment "configs/fraud_sota/elliptic_Naive_HOGRL.yaml"              # IJCAI'24
run_experiment "configs/fraud_sota/elliptic_Naive_CGNN.yaml"                # AAAI'25
run_experiment "configs/fraud_sota/elliptic_Naive_ConsisGAD.yaml"      # ICLR'24
run_experiment "configs/fraud_sota/elliptic_Naive_Grad.yaml"                # WWW'25
run_experiment "configs/fraud_sota/elliptic_Naive_BSL.yaml"                  # AAAI'24 (Sequential)

section "[E] 我们的方法"
run_experiment "configs/ours/main/elliptic_TASDCL_BSL.yaml"                 # TASD-CL (Ours)

# ======================================================================
# ★ PART 2: Elliptic++ Actor Dataset
# ======================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  DATASET 2/3 — Elliptic++ Actor                          ║"
echo "╚══════════════════════════════════════════════════════════╝"

section "[A] 传统静态基线"
run_experiment "configs/traditional/elliptic++actor_Naive_GCN.yaml"
run_experiment "configs/traditional/elliptic++actor_Naive_GAT.yaml"

section "[B] 持续学习经典基线"
run_experiment "configs/cl_baselines/elliptic++actor_EWC_GCN.yaml"
run_experiment "configs/cl_baselines/elliptic++actor_LwF_GCN.yaml"
run_experiment "configs/cl_baselines/elliptic++actor_ER_GCN.yaml"  # ER-GNN

section "[C] 不平衡处理变体"
run_experiment "configs/imbalanced/elliptic++actor_Naive_GraphSMOTE.yaml"

section "[D] 核心 SOTA 欺诈检测模型"
run_experiment "configs/fraud_sota/elliptic++actor_Naive_HOGRL.yaml"
run_experiment "configs/fraud_sota/elliptic++actor_Naive_CGNN.yaml"
run_experiment "configs/fraud_sota/elliptic++actor_Naive_ConsisGAD.yaml"
run_experiment "configs/fraud_sota/elliptic++actor_Naive_Grad.yaml"
run_experiment "configs/PMP/elliptic++actor_Naive_PMP.yaml"
run_experiment "configs/PMP/elliptic++actor_CL_PMP.yaml"
run_experiment "configs/fraud_sota/elliptic++actor_Naive_BSL.yaml"

section "[E] BSL CL 策略对比"

section "[F] 我们的方法 + 消融"
run_experiment "configs/ours/main/elliptic++actor_TASDCL_BSL.yaml"

# ======================================================================
# ★ PART 3: DGraphFin Dataset
# ======================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  DATASET 3/3 — DGraphFin                                 ║"
echo "╚══════════════════════════════════════════════════════════╝"

section "[A] 传统静态基线"
run_experiment "configs/traditional/dgraphfin_Naive_GCN.yaml"
run_experiment "configs/traditional/dgraphfin_Naive_GAT.yaml"

section "[B] 持续学习经典基线"
run_experiment "configs/cl_baselines/dgraphfin_EWC_GCN.yaml"
run_experiment "configs/cl_baselines/dgraphfin_LwF_GCN.yaml"
run_experiment "configs/cl_baselines/dgraphfin_ER_GCN.yaml"

section "[C] 不平衡处理变体"
run_experiment "configs/imbalanced/dgraphfin_Naive_GraphSMOTE.yaml"

section "[D] 核心 SOTA 欺诈检测模型"
run_experiment "configs/fraud_sota/dgraphfin_Naive_HOGRL.yaml"
run_experiment "configs/fraud_sota/dgraphfin_Naive_CGNN.yaml"
run_experiment "configs/fraud_sota/dgraphfin_Naive_ConsisGAD.yaml"
run_experiment "configs/fraud_sota/dgraphfin_Naive_Grad.yaml"
run_experiment "configs/PMP/dgraphfin_Naive_PMP.yaml"
run_experiment "configs/PMP/dgraphfin_CL_PMP.yaml"
run_experiment "configs/fraud_sota/dgraphfin_Naive_BSL.yaml"

section "[E] BSL CL 策略对比"
run_experiment "configs/ours/cl_on_bsl/dgraphfin_EWC_BSL.yaml"
run_experiment "configs/ours/cl_on_bsl/dgraphfin_LwF_BSL.yaml"
run_experiment "configs/ours/cl_on_bsl/dgraphfin_ER_BSL.yaml"

section "[F] 我们的方法 + 消融"
run_experiment "configs/ours/ablation/dgraphfin_TASDCL_noSSF_BSL.yaml"
run_experiment "configs/ours/ablation/dgraphfin_TASDCL_noSPC_BSL.yaml"
run_experiment "configs/ours/ablation/dgraphfin_TASDCL_noSCD_BSL.yaml"
run_experiment "configs/ours/main/dgraphfin_TASDCL_BSL.yaml"

# ======================================================================
# 汇总
# ======================================================================
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
TOTAL_H=$(( TOTAL_ELAPSED / 3600 ))
TOTAL_M=$(( (TOTAL_ELAPSED % 3600) / 60 ))
TOTAL_S=$(( TOTAL_ELAPSED % 60 ))

echo ""
echo "██████████████████████████████████████████████████████████"
echo "█  实验运行汇总                                             █"
printf "█  完成时间: %-46s █\n" "$(date '+%Y-%m-%d %H:%M:%S')"
printf "█  总耗时:   %-46s █\n" "${TOTAL_H}h ${TOTAL_M}m ${TOTAL_S}s"
printf "█  ✅ 已完成: %-2d | ⏭  已跳过: %-2d | ❌ 失败: %-2d          █\n" \
       "${RUN_COUNT}" "${SKIP_COUNT}" "${FAIL_COUNT}"
echo "█                                                          █"
echo "█  运行结果收集:                                            █"
echo "█    python collect_results.py                             █"
echo "██████████████████████████████████████████████████████████"

# 如果有失败则以非0退出，方便 CI 或监控检测
if [[ "${FAIL_COUNT}" -gt 0 ]]; then
    exit 1
fi
exit 0
