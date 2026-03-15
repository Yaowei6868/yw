"""
Elliptic Dataset Experiment Results Analysis Script

Reads all aggregate_metrics CSV files and generates:
1. results_summary.csv - machine-readable summary table
2. results_report.txt  - human-readable comparison table

Handles both CSV formats:
- New format (one row per task): columns avg_f1, avg_macro_f1, avg_auc_roc, etc.
- Old format (single row):       columns macro_f1, macro_auc, g_mean, etc.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

WEIGHTS_DIR = "weights"
OUTPUT_CSV  = "results_summary.csv"
OUTPUT_TXT  = "results_report.txt"

# Maps experiment name to display name and category
EXPERIMENT_REGISTRY = {
    # ── Naive baselines ──────────────────────────────────────────────────────
    "elliptic_Naive_GCN":         ("GCN (Naive)",          "Backbone"),
    "elliptic_Naive_GAT":         ("GAT (Naive)",          "Backbone"),
    "elliptic_Naive_GIN":         ("GIN (Naive)",          "Backbone"),
    "elliptic_Naive_GATv2":       ("GATv2 (Naive)",        "Backbone"),
    "elliptic_Naive_GraphSage":   ("GraphSAGE (Naive)",    "Backbone"),
    "elliptic_Naive_GraphSMOTE":  ("GraphSMOTE (Naive)",   "Backbone"),
    "elliptic_Naive_HOGRL":       ("HOGRL (Naive)",        "Backbone"),
    "elliptic_Naive_CGNN":        ("CGNN (Naive)",         "Backbone"),
    "elliptic_Naive_Grad":        ("Grad (Naive)",         "Backbone"),
    "elliptic_Naive_ConsisGAD":   ("ConsisGAD (Naive)",    "Backbone"),
    "elliptic_Naive_BSL":         ("BSL (Naive)",          "Backbone"),
    # ── GCN CL baselines ────────────────────────────────────────────────────
    "elliptic_EWC_GCN":           ("GCN + EWC",            "CL-GCN"),
    "elliptic_ER_GCN":            ("GCN + ER",             "CL-GCN"),
    "elliptic_LwF_GCN":           ("GCN + LwF",            "CL-GCN"),
    # ── BSL CL baselines ────────────────────────────────────────────────────
    "elliptic_EWC_BSL":           ("BSL + EWC",            "CL-BSL"),
    "elliptic_ER_BSL":            ("BSL + ER",             "CL-BSL"),
    "elliptic_LwF_BSL":           ("BSL + LwF",            "CL-BSL"),
    # ── Ablation study ──────────────────────────────────────────────────────
    "elliptic_TASDCL_noSSF_BSL":  ("TASD-CL w/o SSF",     "Ablation"),
    "elliptic_TASDCL_noSPC_BSL":  ("TASD-CL w/o SPC",     "Ablation"),
    "elliptic_TASDCL_noSCD_BSL":  ("TASD-CL w/o SCD",     "Ablation"),
    # ── Our method ──────────────────────────────────────────────────────────
    "elliptic_TASDCL_BSL":        ("TASD-CL (Ours)",       "Ours"),
    # ── Extended baselines (for supplementary) ───────────────────────────────
    "elliptic_Naive_MLP":         ("MLP (Naive)",          "Extended"),
    "elliptic_Naive_DOMINANT":    ("DOMINANT (Naive)",     "Extended"),
    "elliptic_Naive_EvolveGCN":   ("EvolveGCN (Naive)",   "Extended"),
    "elliptic_Naive_StaGNN":      ("StaGNN (Naive)",       "Extended"),
    "elliptic_Naive_TGN":         ("TGN (Naive)",          "Extended"),
    "elliptic_CL_ConsisGAD":      ("ConsisGAD + CL",       "Extended"),
    "elliptic_CL_GraphSage":      ("GraphSAGE + CL",       "Extended"),
}

CATEGORY_ORDER = ["Backbone", "CL-GCN", "CL-BSL", "Ablation", "Ours", "Extended"]


# ─── CSV Parsing ─────────────────────────────────────────────────────────────

def parse_new_format(df: pd.DataFrame, exp_name: str) -> dict | None:
    """New format: one row per task, columns avg_f1, avg_macro_f1, etc."""
    last_task = df.iloc[-1]
    num_tasks = len(df)

    # Forgetting: avg metric at task 1 - avg metric at last task
    f1_seq     = df["avg_f1"].values
    macro_f1   = df["avg_macro_f1"].values
    auc_roc    = df["avg_auc_roc"].values
    g_mean     = df["avg_g_mean"].values

    # Backward Transfer (BWT) / Forgetting
    # Using the standard definition: average drop in F1 from task i's best to final
    # Simple proxy: last_task_macro_f1 vs first_task_macro_f1
    forgetting = float(macro_f1[0] - macro_f1[-1]) if num_tasks > 1 else 0.0

    return {
        "exp_name":       exp_name,
        "num_tasks":      num_tasks,
        "final_avg_f1":       float(last_task["avg_f1"]),
        "final_macro_f1":     float(last_task["avg_macro_f1"]),
        "final_auc_roc":      float(last_task["avg_auc_roc"]),
        "final_g_mean":       float(last_task["avg_g_mean"]),
        "mean_avg_f1":        float(f1_seq.mean()),
        "mean_macro_f1":      float(macro_f1.mean()),
        "mean_auc_roc":       float(auc_roc.mean()),
        "mean_g_mean":        float(g_mean.mean()),
        "forgetting":         forgetting,
        "cl_mode":            str(last_task.get("cl_mode", "Unknown")),
        "format":             "new",
    }


def parse_old_format(df: pd.DataFrame, exp_name: str) -> dict | None:
    """Old format: single summary row, columns macro_f1, macro_auc, g_mean."""
    row = df.iloc[0]
    macro_f1 = float(row.get("macro_f1", 0.0))
    auc      = float(row.get("macro_auc", 0.0))
    g_mean   = float(row.get("g_mean", 0.0))
    cl_mode  = str(row.get("cl_mode", "Unknown"))

    return {
        "exp_name":       exp_name,
        "num_tasks":      1,
        "final_avg_f1":       macro_f1,
        "final_macro_f1":     macro_f1,
        "final_auc_roc":      auc,
        "final_g_mean":       g_mean,
        "mean_avg_f1":        macro_f1,
        "mean_macro_f1":      macro_f1,
        "mean_auc_roc":       auc,
        "mean_g_mean":        g_mean,
        "forgetting":         float(row.get("avg_recall_forgetting", 0.0)),
        "cl_mode":            cl_mode,
        "format":             "old",
    }


def parse_f1macro_format(df: pd.DataFrame, exp_name: str) -> dict | None:
    """
    Third format: columns f1_macro, auc_roc, g_mean (used by ConsisGAD CL,
    FraudGNN_RL, old Replaygraphsage). May have multiple rows (one per task).
    """
    last_task = df.iloc[-1]
    num_tasks = len(df)
    macro_f1_col = df["f1_macro"].values
    auc_col      = df["auc_roc"].values if "auc_roc" in df.columns else np.zeros(num_tasks)
    gm_col       = df["g_mean"].values  if "g_mean"  in df.columns else np.zeros(num_tasks)

    # 'forgetting' may already be a column
    if "forgetting" in df.columns:
        forgetting = float(last_task["forgetting"])
    else:
        forgetting = float(macro_f1_col[0] - macro_f1_col[-1]) if num_tasks > 1 else 0.0

    return {
        "exp_name":       exp_name,
        "num_tasks":      num_tasks,
        "final_avg_f1":       float(last_task["f1_macro"]),
        "final_macro_f1":     float(last_task["f1_macro"]),
        "final_auc_roc":      float(auc_col[-1]),
        "final_g_mean":       float(gm_col[-1]),
        "mean_avg_f1":        float(macro_f1_col.mean()),
        "mean_macro_f1":      float(macro_f1_col.mean()),
        "mean_auc_roc":       float(auc_col.mean()),
        "mean_g_mean":        float(gm_col.mean()),
        "forgetting":         forgetting,
        "cl_mode":            str(last_task.get("cl_mode", "Unknown")),
        "format":             "new",
    }


def load_experiment_metrics(csv_path: str) -> dict | None:
    """Load and parse a single aggregate_metrics CSV file."""
    exp_name = Path(csv_path).stem.replace("_aggregate_metrics", "")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [WARN] Cannot read {csv_path}: {e}")
        return None

    if df.empty:
        return None

    cols = set(df.columns)
    # Detect format by column names
    if "avg_f1" in cols:
        return parse_new_format(df, exp_name)
    elif "macro_f1" in cols:
        return parse_old_format(df, exp_name)
    elif "f1_macro" in cols:
        return parse_f1macro_format(df, exp_name)
    else:
        print(f"  [WARN] Unknown format in {csv_path}, columns: {list(df.columns)}")
        return None


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Elliptic Dataset - Experiment Results Analysis")
    print("=" * 60)

    # Find all aggregate_metrics CSV files
    csv_files = glob.glob(
        os.path.join(WEIGHTS_DIR, "**", "*_aggregate_metrics.csv"),
        recursive=True
    )
    csv_files = [f for f in csv_files if "elliptic" in Path(f).stem
                 and "actor" not in Path(f).stem]  # skip elliptic++ actor
    csv_files.sort()

    print(f"\nFound {len(csv_files)} CSV files (elliptic only, excluding actor):\n")

    # Load all results
    results = []
    for csv_path in csv_files:
        exp_name = Path(csv_path).stem.replace("_aggregate_metrics", "")
        metrics  = load_experiment_metrics(csv_path)
        if metrics:
            results.append(metrics)
            fmt_tag = "[new]" if metrics["format"] == "new" else "[old]"
            print(f"  {fmt_tag} {exp_name}")
            print(f"        final macro_f1={metrics['final_macro_f1']:.4f}  "
                  f"auc={metrics['final_auc_roc']:.4f}  "
                  f"g_mean={metrics['final_g_mean']:.4f}  "
                  f"forgetting={metrics['forgetting']:+.4f}")

    if not results:
        print("\n[ERROR] No results found.")
        return

    # Build summary DataFrame
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved machine-readable summary: {OUTPUT_CSV}")

    # ─── Generate Human-Readable Report ──────────────────────────────────────
    lines = []
    lines.append("=" * 90)
    lines.append("  FRAUD DETECTION ON ELLIPTIC DATASET - RESULTS SUMMARY")
    lines.append("=" * 90)
    lines.append(
        f"  {'Method':<30} {'Category':<12} "
        f"{'Macro-F1':>10} {'AUC-ROC':>10} {'G-Mean':>10} "
        f"{'Avg-F1':>10} {'Forgetting':>12} {'Format':<6}"
    )
    lines.append("-" * 90)

    for category in CATEGORY_ORDER:
        category_rows = []

        # Check registry first, then all results
        for exp_name, (display, cat) in EXPERIMENT_REGISTRY.items():
            if cat != category:
                continue
            row = df_results[df_results["exp_name"] == exp_name]
            if row.empty:
                category_rows.append(
                    f"  {'* ' + display:<30} {cat:<12} "
                    f"{'—':>10} {'—':>10} {'—':>10} {'—':>10} {'—':>12} {'N/A':<6}"
                )
            else:
                r = row.iloc[0]
                forgetting_str = f"{r['forgetting']:+.4f}"
                category_rows.append(
                    f"  {display:<30} {cat:<12} "
                    f"{r['final_macro_f1']:>10.4f} {r['final_auc_roc']:>10.4f} "
                    f"{r['final_g_mean']:>10.4f} {r['mean_avg_f1']:>10.4f} "
                    f"{forgetting_str:>12} {r['format']:<6}"
                )

        if category_rows:
            lines.extend(category_rows)
            lines.append("-" * 90)

    # Also add results not in registry
    known_names = set(EXPERIMENT_REGISTRY.keys())
    extra_rows = df_results[~df_results["exp_name"].isin(known_names)]
    if not extra_rows.empty:
        lines.append("  [Other experiments not in registry]")
        for _, r in extra_rows.iterrows():
            lines.append(
                f"  {r['exp_name']:<30} {'Other':<12} "
                f"{r['final_macro_f1']:>10.4f} {r['final_auc_roc']:>10.4f} "
                f"{r['final_g_mean']:>10.4f} {r['mean_avg_f1']:>10.4f} "
                f"{r['forgetting']:>+12.4f} {r['format']:<6}"
            )
        lines.append("-" * 90)

    lines.append("")
    lines.append("  (* = experiment not yet run)")
    lines.append("  Metrics measured at final task (task 10) over cumulative test set")
    lines.append("  Forgetting = macro_f1[task1] - macro_f1[task10] (positive = catastrophic forgetting)")
    lines.append("")

    # ─── Best Results Highlight ───────────────────────────────────────────────
    lines.append("=" * 90)
    lines.append("  HIGHLIGHTS")
    lines.append("=" * 90)

    new_fmt = df_results[df_results["format"] == "new"]
    if not new_fmt.empty:
        best_f1  = new_fmt.loc[new_fmt["final_macro_f1"].idxmax()]
        best_auc = new_fmt.loc[new_fmt["final_auc_roc"].idxmax()]
        best_gm  = new_fmt.loc[new_fmt["final_g_mean"].idxmax()]
        min_fg   = new_fmt.loc[new_fmt["forgetting"].idxmin()]

        lines.append(
            f"  Best Macro-F1:  {best_f1['exp_name']:<35} "
            f"{best_f1['final_macro_f1']:.4f}"
        )
        lines.append(
            f"  Best AUC-ROC:   {best_auc['exp_name']:<35} "
            f"{best_auc['final_auc_roc']:.4f}"
        )
        lines.append(
            f"  Best G-Mean:    {best_gm['exp_name']:<35} "
            f"{best_gm['final_g_mean']:.4f}"
        )
        lines.append(
            f"  Least Forgetting: {min_fg['exp_name']:<33} "
            f"{min_fg['forgetting']:+.4f}"
        )

        # TASD-CL specific comparison
        tasdcl = df_results[df_results["exp_name"] == "elliptic_TASDCL_BSL"]
        bsl    = df_results[df_results["exp_name"] == "elliptic_Naive_BSL"]
        if not tasdcl.empty and not bsl.empty:
            t = tasdcl.iloc[0]
            b = bsl.iloc[0]
            lines.append("")
            lines.append("  TASD-CL vs BSL Naive:")
            lines.append(
                f"    Macro-F1:  {b['final_macro_f1']:.4f} → {t['final_macro_f1']:.4f} "
                f"({t['final_macro_f1'] - b['final_macro_f1']:+.4f})"
            )
            lines.append(
                f"    AUC-ROC:   {b['final_auc_roc']:.4f} → {t['final_auc_roc']:.4f} "
                f"({t['final_auc_roc'] - b['final_auc_roc']:+.4f})"
            )
            lines.append(
                f"    Forgetting:{b['forgetting']:+.4f} → {t['forgetting']:+.4f} "
                f"({t['forgetting'] - b['forgetting']:+.4f})"
            )

    lines.append("")
    report = "\n".join(lines)
    print("\n" + report)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved human-readable report: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
