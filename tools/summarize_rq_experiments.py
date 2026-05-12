"""
Summarize RQ3/RQ4/hyperparameter experiment CSV files.

Usage:
    python tools/summarize_rq_experiments.py --kind robustness
    python tools/summarize_rq_experiments.py --kind efficiency
    python tools/summarize_rq_experiments.py --kind hyperparam
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOTS = {
    "robustness": Path("weights/robustness"),
    "efficiency": Path("weights/efficiency"),
    "hyperparam": Path("weights/hyperparam"),
}


def load_final_rows(root: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(root.glob("**/metrics/*_aggregate_metrics.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] skip {csv_path}: {exc}")
            continue
        if df.empty:
            continue
        final = df.sort_values("task_id").iloc[-1].to_dict()
        final["exp_name"] = csv_path.stem.replace("_aggregate_metrics", "")
        final["csv_path"] = str(csv_path)
        rows.append(final)
    return pd.DataFrame(rows)


def summarize(kind: str, out_dir: Path) -> None:
    root = ROOTS[kind]
    if not root.exists():
        print(f"[INFO] {root} does not exist. Run the experiment script first.")
        return

    df = load_final_rows(root)
    if df.empty:
        print(f"[INFO] no completed CSV files under {root}")
        return

    metric_cols = [
        "exp_name",
        "task_id",
        "avg_auc_roc",
        "avg_macro_f1",
        "avg_f1",
        "avg_g_mean",
        "time_cost",
        "avg_inference_time_ms",
        "peak_gpu_memory_mb",
        "model_param_mb",
        "stored_memory_mb",
        "csv_path",
    ]
    cols = [c for c in metric_cols if c in df.columns]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{kind}_summary.csv"
    df[cols].sort_values("exp_name").to_csv(out_path, index=False, float_format="%.6f")
    print(f"[SAVED] {out_path}")

    display_cols = [c for c in cols if c not in {"csv_path"}]
    print(df[display_cols].sort_values("exp_name").to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=sorted(ROOTS),
        required=True,
        help="Experiment family to summarize.",
    )
    parser.add_argument("--out", default="paper_results", help="Output directory.")
    args = parser.parse_args()
    summarize(args.kind, Path(args.out))


if __name__ == "__main__":
    main()
