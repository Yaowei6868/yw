"""
collect_results.py
==================
收集所有实验 CSV，生成论文用汇总表。

用法:
    python collect_results.py                   # 默认输出到 paper_results/
    python collect_results.py --out my_results  # 自定义输出目录
    python collect_results.py --latex           # 同时输出 LaTeX 表格
"""

import os
import glob
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# 实验名称 -> (论文显示名, 数据集, 分组)
# 分组用于 LaTeX 表格中的 \midrule 分隔
# =============================================================================
METHOD_MAP = {
    # ── Elliptic ──────────────────────────────────────────────────────────
    "elliptic_Naive_GCN":           ("GCN (Sequential)",    "Elliptic",  "A-Static"),
    "elliptic_Naive_GAT":           ("GAT (Sequential)",    "Elliptic",  "A-Static"),
    "elliptic_EWC_GCN":             ("GCN + EWC",           "Elliptic",  "B-CL"),
    "elliptic_LwF_GCN":             ("GCN + LwF",           "Elliptic",  "B-CL"),
    "elliptic_ER_GCN":              ("GCN + ER (ER-GNN)",   "Elliptic",  "B-CL"),
    "elliptic_Naive_GraphSMOTE":    ("GCN + GraphSMOTE",    "Elliptic",  "C-Imbalance"),
    "elliptic_Naive_HOGRL":         ("HOGRL (IJCAI'24)",    "Elliptic",  "D-SOTA"),
    "elliptic_Naive_CGNN":          ("CGNN (AAAI'25)",      "Elliptic",  "D-SOTA"),
    "elliptic_Naive_ConsisGAD":     ("ConsisGAD (ICLR'24)", "Elliptic",  "D-SOTA"),
    "elliptic_Naive_Grad":          ("GradGNN (WWW'25)",    "Elliptic",  "D-SOTA"),
    "elliptic_Naive_BSL":           ("BSL (AAAI'24, Seq.)", "Elliptic",  "D-SOTA"),
    "elliptic_TASDCL_BSL":          ("TASD-CL (Ours)",      "Elliptic",  "E-Ours"),

    # ── Elliptic++ Actor ──────────────────────────────────────────────────
    "elliptic_actor_Naive_GCN":         ("GCN (Sequential)",    "Elliptic++", "A-Static"),
    "elliptic_actor_Naive_GAT":         ("GAT (Sequential)",    "Elliptic++", "A-Static"),
    "elliptic_actor_EWC_GCN":           ("GCN + EWC",           "Elliptic++", "B-CL"),
    "elliptic_actor_LwF_GCN":           ("GCN + LwF",           "Elliptic++", "B-CL"),
    "elliptic_actor_ER_GCN":            ("GCN + ER (ER-GNN)",   "Elliptic++", "B-CL"),
    "elliptic_actor_Naive_GraphSMOTE":  ("GCN + GraphSMOTE",    "Elliptic++", "C-Imbalance"),
    "elliptic_actor_Naive_HOGRL":       ("HOGRL (IJCAI'24)",    "Elliptic++", "D-SOTA"),
    "elliptic_actor_Naive_CGNN":        ("CGNN (AAAI'25)",      "Elliptic++", "D-SOTA"),
    "elliptic_actor_Naive_ConsisGAD":   ("ConsisGAD (ICLR'24)", "Elliptic++", "D-SOTA"),
    "elliptic_actor_Naive_Grad":        ("GradGNN (WWW'25)",    "Elliptic++", "D-SOTA"),
    "elliptic_actor_Naive_BSL":         ("BSL (AAAI'24, Seq.)", "Elliptic++", "D-SOTA"),
    "elliptic_actor_TASDCL_BSL":        ("TASD-CL (Ours)",      "Elliptic++", "E-Ours"),
}

# 最终汇总表用到的指标列 (CSV 中的原始列名)
METRIC_COLS = [
    "avg_f1",        # F1 (二分类欺诈类)
    "avg_auc_roc",   # AUC-ROC
    "avg_auc_pr",    # AUC-PR (AP)
    "avg_g_mean",    # G-Mean
    "avg_forgetting", # 平均遗忘率 (↓)
    "avg_bwt",       # 后向迁移 BWT (↑)
]

# 论文显示名
METRIC_DISPLAY = {
    "avg_f1":         "F1 (↑)",
    "avg_auc_roc":    "AUC-ROC (↑)",
    "avg_auc_pr":     "AUC-PR (↑)",
    "avg_g_mean":     "G-Mean (↑)",
    "avg_forgetting": "Forgetting (↓)",
    "avg_bwt":        "BWT (↑)",
}

# 论文中方法出现的顺序
METHOD_ORDER = [
    "GCN (Sequential)",
    "GAT (Sequential)",
    "GCN + EWC",
    "GCN + LwF",
    "GCN + ER (ER-GNN)",
    "GCN + GraphSMOTE",
    "HOGRL (IJCAI'24)",
    "CGNN (AAAI'25)",
    "ConsisGAD (ICLR'24)",
    "GradGNN (WWW'25)",
    "BSL (AAAI'24, Seq.)",
    "TASD-CL (Ours)",
]


# =============================================================================
# 核心函数
# =============================================================================

def find_all_csvs(weights_root: str) -> list[Path]:
    """递归搜索所有 *_aggregate_metrics.csv"""
    pattern = os.path.join(weights_root, "**", "metrics", "*_aggregate_metrics.csv")
    return [Path(p) for p in glob.glob(pattern, recursive=True)]


def load_experiment(csv_path: Path) -> dict | None:
    """
    读取一个 CSV，返回最终任务行的指标字典。
    同时附带每个任务的 F1（用于作图）。
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [WARN] 无法读取 {csv_path}: {e}")
        return None

    if df.empty:
        return None

    # 推断实验名 (从文件名去掉 _aggregate_metrics 后缀)
    exp_name = csv_path.stem.replace("_aggregate_metrics", "")

    # 最终行 (task_id 最大的那行 = 所有任务训练完毕后的聚合指标)
    final_row = df.sort_values("task_id").iloc[-1]

    result = {"exp_name": exp_name, "csv_path": str(csv_path), "num_tasks": len(df)}
    for col in METRIC_COLS:
        result[col] = final_row.get(col, np.nan)

    # 保存每任务 F1 序列（用于折线图）
    result["f1_series"] = df.sort_values("task_id")["avg_f1"].tolist()

    return result


def build_summary_df(records: list[dict]) -> pd.DataFrame:
    """将所有实验记录整理为宽表"""
    rows = []
    for r in records:
        exp = r["exp_name"]
        if exp not in METHOD_MAP:
            continue
        method_name, dataset, group = METHOD_MAP[exp]
        row = {
            "Method": method_name,
            "Dataset": dataset,
            "Group": group,
            "Exp_Name": exp,
            "Num_Tasks": r["num_tasks"],
        }
        for col in METRIC_COLS:
            row[METRIC_DISPLAY[col]] = r.get(col, np.nan)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # 按数据集 + 分组 + 方法顺序排序
    df["_order"] = df["Method"].map(
        {m: i for i, m in enumerate(METHOD_ORDER)}
    ).fillna(999)
    df = df.sort_values(["Dataset", "Group", "_order"]).drop(columns=["_order"])
    return df.reset_index(drop=True)


def print_table(summary_df: pd.DataFrame):
    """在终端打印格式化结果表"""
    if summary_df.empty:
        print("  [INFO] 暂无已完成的实验结果。")
        return

    metric_cols = list(METRIC_DISPLAY.values())

    for dataset in ["Elliptic", "Elliptic++"]:
        sub = summary_df[summary_df["Dataset"] == dataset]
        if sub.empty:
            continue

        print(f"\n{'='*80}")
        print(f"  Dataset: {dataset}")
        print(f"{'='*80}")

        header = f"  {'Method':<28}" + "".join(f"{c:>14}" for c in metric_cols)
        print(header)
        print(f"  {'-'*28}" + "".join(f"{'-'*14}" for _ in metric_cols))

        last_group = None
        for _, row in sub.iterrows():
            if last_group is not None and row["Group"] != last_group:
                print(f"  {'·'*28}" + "".join(f"{'·'*14}" for _ in metric_cols))
            last_group = row["Group"]

            marker = " ★" if "Ours" in row["Method"] else "  "
            line = f"{marker}{row['Method']:<26}"
            for col in metric_cols:
                val = row.get(col, np.nan)
                if pd.isna(val):
                    line += f"{'  —':>14}"
                else:
                    line += f"{val:>14.4f}"
            print(line)


def print_missing(summary_df: pd.DataFrame):
    """列出尚未完成的实验"""
    completed = set(summary_df["Exp_Name"].tolist()) if not summary_df.empty else set()
    missing = [exp for exp in METHOD_MAP if exp not in completed]
    if not missing:
        print("\n  ✅ 所有实验已完成！")
        return

    print(f"\n  ⏳ 尚未完成的实验 ({len(missing)}/{len(METHOD_MAP)}):")
    for exp in missing:
        method, dataset, _ = METHOD_MAP[exp]
        print(f"    - [{dataset}] {method}  (exp: {exp})")


def save_csv(summary_df: pd.DataFrame, out_dir: str):
    """保存汇总 CSV"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results_summary.csv")
    summary_df.to_csv(path, index=False, float_format="%.4f")
    print(f"\n  [SAVED] 汇总 CSV → {path}")
    return path


def save_pertask_csv(records: list[dict], out_dir: str):
    """保存每任务 F1 时间序列（用于折线图）"""
    rows = []
    for r in records:
        exp = r["exp_name"]
        if exp not in METHOD_MAP:
            continue
        method_name, dataset, _ = METHOD_MAP[exp]
        for t, f1 in enumerate(r.get("f1_series", []), start=1):
            rows.append({
                "Dataset": dataset,
                "Method": method_name,
                "Exp_Name": exp,
                "Task": t,
                "Avg_F1": f1,
            })
    if rows:
        path = os.path.join(out_dir, "results_pertask_f1.csv")
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4f")
        print(f"  [SAVED] 每任务 F1 序列 → {path}")


def save_latex(summary_df: pd.DataFrame, out_dir: str):
    """生成 LaTeX booktabs 表格（Elliptic + Elliptic++ 各一张）"""
    if summary_df.empty:
        return

    metric_cols_display = list(METRIC_DISPLAY.values())

    for dataset in ["Elliptic", "Elliptic++"]:
        sub = summary_df[summary_df["Dataset"] == dataset].copy()
        if sub.empty:
            continue

        lines = []
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{Main results on " + dataset + r" dataset. "
            r"Best results in \textbf{bold}, second-best \underline{underlined}.}"
        )
        col_spec = "l" + "r" * len(metric_cols_display)
        lines.append(r"\begin{tabular}{" + col_spec + r"}")
        lines.append(r"\toprule")

        # 表头
        header_cells = ["Method"] + metric_cols_display
        lines.append(" & ".join(header_cells) + r" \\")
        lines.append(r"\midrule")

        # 计算每列最优、次优（仅数值列）
        best_vals = {}
        second_vals = {}
        for col in metric_cols_display:
            vals = sub[col].dropna().values
            if len(vals) == 0:
                continue
            # Forgetting: 越小越好
            if "Forgetting" in col:
                sorted_v = sorted(set(vals))
                best_vals[col] = sorted_v[0] if len(sorted_v) > 0 else None
                second_vals[col] = sorted_v[1] if len(sorted_v) > 1 else None
            else:
                sorted_v = sorted(set(vals), reverse=True)
                best_vals[col] = sorted_v[0] if len(sorted_v) > 0 else None
                second_vals[col] = sorted_v[1] if len(sorted_v) > 1 else None

        last_group = None
        for _, row in sub.iterrows():
            if last_group is not None and row["Group"] != last_group:
                lines.append(r"\midrule")
            last_group = row["Group"]

            is_ours = "Ours" in row["Method"]
            method_cell = r"\textit{" + row["Method"] + r"}" if is_ours else row["Method"]
            cells = [method_cell]

            for col in metric_cols_display:
                val = row.get(col, np.nan)
                if pd.isna(val):
                    cells.append("—")
                    continue
                fmt = f"{val:.4f}"
                if best_vals.get(col) is not None and abs(val - best_vals[col]) < 1e-7:
                    fmt = r"\textbf{" + fmt + r"}"
                elif second_vals.get(col) is not None and abs(val - second_vals[col]) < 1e-7:
                    fmt = r"\underline{" + fmt + r"}"
                cells.append(fmt)

            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table*}")

        fname = f"latex_table_{dataset.replace('+', 'plus').replace(' ', '_')}.tex"
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  [SAVED] LaTeX 表格 → {path}")


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="收集实验结果并生成论文表格")
    parser.add_argument("--weights", default="weights",
                        help="weights 根目录 (默认: weights)")
    parser.add_argument("--out", default="paper_results",
                        help="输出目录 (默认: paper_results)")
    parser.add_argument("--latex", action="store_true",
                        help="同时生成 LaTeX 表格文件")
    args = parser.parse_args()

    # 切换到脚本所在目录，保证相对路径正确
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    print("\n" + "=" * 60)
    print("  collect_results.py — 实验结果收集器")
    print("=" * 60)
    print(f"  搜索目录: {args.weights}/")

    # 1. 找到所有 CSV
    csvs = find_all_csvs(args.weights)
    print(f"  找到 CSV 文件: {len(csvs)} 个")

    # 2. 加载实验记录
    records = []
    for csv_path in sorted(csvs):
        r = load_experiment(csv_path)
        if r is not None:
            records.append(r)
            status = "✅" if r["num_tasks"] >= 10 else f"⚠ ({r['num_tasks']}/10 tasks)"
            print(f"    {status}  {r['exp_name']}")

    if not records:
        print("\n  [INFO] 未找到任何结果，请先运行实验。")
        return

    # 3. 构建汇总 DataFrame
    summary_df = build_summary_df(records)

    # 4. 打印结果表
    print_table(summary_df)

    # 5. 列出缺失实验
    print_missing(summary_df)

    # 6. 保存文件
    print()
    save_csv(summary_df, args.out)
    save_pertask_csv(records, args.out)
    if args.latex:
        save_latex(summary_df, args.out)

    # 7. 完成度统计
    completed = len(summary_df["Exp_Name"].unique()) if not summary_df.empty else 0
    total = len(METHOD_MAP)
    print(f"\n  进度: {completed}/{total} 实验已完成 ({100*completed//total}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
