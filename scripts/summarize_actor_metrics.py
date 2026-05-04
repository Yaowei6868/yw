from __future__ import annotations

import csv
from pathlib import Path


FILES = [
    ("TASD-CL current", "weights/cgnn/elliptic_actor_TASDCL_CGNN/metrics/elliptic_actor_TASDCL_CGNN_aggregate_metrics.csv"),
    ("CGNN", "weights/cgnn/elliptic_actor_Naive_CGNN/metrics/elliptic_actor_Naive_CGNN_aggregate_metrics.csv"),
    ("HOGRL", "weights/hogrl/elliptic_actor_Naive_HOGRL/metrics/elliptic_actor_Naive_HOGRL_aggregate_metrics.csv"),
    ("GradGNN", "weights/grad/elliptic_actor_Naive_Grad/metrics/elliptic_actor_Naive_Grad_aggregate_metrics.csv"),
    ("BSL", "weights/bsl/elliptic_actor_Naive_BSL/metrics/elliptic_actor_Naive_BSL_aggregate_metrics.csv"),
    ("GCN", "weights/gcn/elliptic_actor_Naive_GCN/metrics/elliptic_actor_Naive_GCN_aggregate_metrics.csv"),
    ("GCN + EWC", "weights/gcn/elliptic_actor_EWC_GCN/metrics/elliptic_actor_EWC_GCN_aggregate_metrics.csv"),
    ("GCN + LwF", "weights/gcn/elliptic_actor_LwF_GCN/metrics/elliptic_actor_LwF_GCN_aggregate_metrics.csv"),
    ("GCN + ER", "weights/gcn/elliptic_actor_ER_GCN/metrics/elliptic_actor_ER_GCN_aggregate_metrics.csv"),
]


def iter_actor_opt_files() -> list[tuple[str, str]]:
    files = []
    root = Path("weights/tasd")
    if root.exists():
        for path in sorted(root.glob("actor_TASDCL_*/metrics/*_aggregate_metrics.csv")):
            files.append((path.parts[-3], str(path)))

    root = Path("weights/baseline_opt")
    if root.exists():
        for path in sorted(root.glob("actor_*/metrics/*_aggregate_metrics.csv")):
            files.append((path.parts[-3], str(path)))

    return files


def read_last(path: str) -> dict[str, str] | None:
    p = Path(path)
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open(newline="")))
    return rows[-1] if rows else None


def fmt(value: str) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "NA"


def main() -> None:
    rows = []
    for name, path in FILES + iter_actor_opt_files():
        row = read_last(path)
        if row is None:
            rows.append((name, "MISSING", "MISSING", "MISSING", "MISSING"))
            continue
        rows.append((
            name,
            fmt(row.get("avg_auc_roc", "")),
            fmt(row.get("avg_g_mean", "")),
            fmt(row.get("avg_macro_f1", "")),
            fmt(row.get("avg_f1", "")),
        ))

    print("| Method | AUC-ROC | G-Mean | MacroF1 | F1 |")
    print("|---|---:|---:|---:|---:|")
    for name, auc, gmean, macro, f1 in rows:
        print(f"| {name} | {auc} | {gmean} | {macro} | {f1} |")

    numeric = []
    for name, auc, gmean, macro, _ in rows:
        try:
            numeric.append((name, float(auc), float(gmean), float(macro)))
        except ValueError:
            pass
    if numeric:
        print()
        print("Best by metric:")
        print(f"- AUC-ROC: {max(numeric, key=lambda x: x[1])[0]}")
        print(f"- G-Mean: {max(numeric, key=lambda x: x[2])[0]}")
        print(f"- MacroF1: {max(numeric, key=lambda x: x[3])[0]}")


if __name__ == "__main__":
    main()
