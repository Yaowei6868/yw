#!/bin/bash
# run_tasdcl_rerun.sh
# 重跑 TASD-CL 全量 + 三个消融实验（修复 warm-up bug 后的版本）
#
# 用法（服务器）：
#   bash scripts/run_tasdcl_rerun.sh          # 前台，可直接看进度
#   nohup bash scripts/run_tasdcl_rerun.sh \
#       > logs/tasdcl_rerun.log 2>&1 &        # 后台，用 tail -f 跟踪

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PYTHON="${PYTHON:-python}"

mkdir -p "${LOG_DIR}"

NUM_TASKS=10
COMPLETE_ROWS=$((NUM_TASKS + 1))   # header + 10 task rows

SKIP_COUNT=0
RUN_COUNT=0
FAIL_COUNT=0
TOTAL_START=$(date +%s)

# ── 工具函数 ──────────────────────────────────────────────────────────────

_yaml_field() {
    grep -m1 "^${2}:" "$1" | awk '{print $2}' | tr -d '"' | tr -d "'"
}

is_complete() {
    local config_path="$1"
    local exp_name save_dir csv_path row_count
    exp_name=$(_yaml_field "$config_path" "name")
    save_dir=$(grep -m1 "save_dir:" "$config_path" | awk '{print $2}' | tr -d '"' | tr -d "'")
    csv_path="${ROOT_DIR}/${save_dir}/metrics/${exp_name}_aggregate_metrics.csv"
    if [[ -f "$csv_path" ]]; then
        row_count=$(wc -l < "$csv_path" 2>/dev/null || echo 0)
        if [[ "$row_count" -ge "$COMPLETE_ROWS" ]]; then
            return 0   # complete → skip
        fi
    fi
    return 1   # not complete → run
}

run_experiment() {
    local config_path="$1"
    local label="$2"
    local exp_name log_file t_start t_end elapsed

    exp_name=$(_yaml_field "$config_path" "name")
    log_file="${LOG_DIR}/${exp_name}.log"

    echo ""
    echo "  ┌─────────────────────────────────────────────────────────┐"
    printf  "  │  %s: %-50s│\n" "${label}" "${exp_name}"
    echo "  └─────────────────────────────────────────────────────────┘"

    if is_complete "$config_path"; then
        echo "  [SKIP] 结果已完整 (${COMPLETE_ROWS} 行 CSV)，跳过"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return
    fi

    echo "  [RUN ] 开始训练 @ $(date '+%H:%M:%S')"
    echo "         日志: logs/${exp_name}.log"
    echo "         实时查看: tail -f logs/${exp_name}.log"

    t_start=$(date +%s)
    cd "${ROOT_DIR}"

    if ${PYTHON} train.py --config "${config_path}" > "${log_file}" 2>&1; then
        t_end=$(date +%s)
        elapsed=$(( t_end - t_start ))
        printf "  [DONE] %-48s %dm%02ds\n" \
               "${exp_name}" $(( elapsed / 60 )) $(( elapsed % 60 ))
        RUN_COUNT=$((RUN_COUNT + 1))
    else
        echo "  [FAIL] ${exp_name}"
        echo "  最后 10 行日志："
        tail -10 "${log_file}" | sed 's/^/    /'
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# ── 主流程 ────────────────────────────────────────────────────────────────

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  TASD-CL 重跑脚本（warm-up bug 修复后）                  ║"
printf "║  开始时间: %-46s║\n" "$(date '+%Y-%m-%d %H:%M:%S')"
echo "╠═══════════════════════════════════════════════════════════╣"
echo "║  实验列表:                                                ║"
echo "║    1. elliptic_TASDCL_BSL         (full TASD-CL)         ║"
echo "║    2. elliptic_TASDCL_noSSF_BSL   (w/o SSF)              ║"
echo "║    3. elliptic_TASDCL_noSPC_BSL   (w/o SPC)              ║"
echo "║    4. elliptic_TASDCL_noSCD_BSL   (w/o SCD)              ║"
echo "╚═══════════════════════════════════════════════════════════╝"

echo ""
echo "  ── [1/4] Full TASD-CL (SSF + SPC + SCD) ──────────────────"
run_experiment "configs/ours/main/elliptic_TASDCL_BSL.yaml" "1/4"

echo ""
echo "  ── [2/4] Ablation: w/o SSF (SPC + SCD only) ───────────────"
run_experiment "configs/ours/ablation/elliptic_TASDCL_noSSF_BSL.yaml" "2/4"

echo ""
echo "  ── [3/4] Ablation: w/o SPC (SSF + SCD only) ───────────────"
run_experiment "configs/ours/ablation/elliptic_TASDCL_noSPC_BSL.yaml" "3/4"

echo ""
echo "  ── [4/4] Ablation: w/o SCD (SSF + SPC only) ───────────────"
run_experiment "configs/ours/ablation/elliptic_TASDCL_noSCD_BSL.yaml" "4/4"

# ── 汇总 ─────────────────────────────────────────────────────────────────

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  运行完毕                                                 ║"
printf "║  结束时间: %-46s║\n" "$(date '+%Y-%m-%d %H:%M:%S')"
printf "║  总耗时:   %dh %dm %ds%-38s║\n" \
       $(( TOTAL_ELAPSED / 3600 )) \
       $(( (TOTAL_ELAPSED % 3600) / 60 )) \
       $(( TOTAL_ELAPSED % 60 )) ""
echo "╠═══════════════════════════════════════════════════════════╣"
printf "║  ✅ 成功运行: %-2d   ⏭  跳过: %-2d   ❌ 失败: %-2d          ║\n" \
       "${RUN_COUNT}" "${SKIP_COUNT}" "${FAIL_COUNT}"
echo "╠═══════════════════════════════════════════════════════════╣"
echo "║  下一步: git add weights && git push                     ║"
echo "╚═══════════════════════════════════════════════════════════╝"

[[ "${FAIL_COUNT}" -gt 0 ]] && exit 1
exit 0
