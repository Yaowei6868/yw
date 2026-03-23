"""
check_threshold.py
------------------
加载已训练好的模型，检查输出概率分布，
对比 threshold=0.3/0.4/0.5 下各模型的 F1/Precision/Recall，
帮助选定最终评估阈值。

用法（在服务器项目根目录下运行）：
    python tools/check_threshold.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from fraud_detection.datasets import get_dataset
from fraud_detection.models import get_model

# ── 要检查的实验列表 (config_path, weight_path) ──────────────────────────────
EXPERIMENTS = [
    ("configs/fraud_sota/elliptic_Naive_BSL.yaml",
     "weights/bsl/elliptic_Naive_BSL/elliptic_Naive_BSL.pt"),
    ("configs/fraud_sota/elliptic_Naive_HOGRL.yaml",
     "weights/hogrl/elliptic_Naive_HOGRL.pt"),
    ("configs/fraud_sota/elliptic_Naive_CGNN.yaml",
     "weights/cgnn/elliptic_Naive_CGNN.pt"),
    ("configs/fraud_sota/elliptic_Naive_Grad.yaml",
     "weights/grad/elliptic_Naive_Grad.pt"),
    ("configs/fraud_sota/elliptic_Naive_ConsisGAD.yaml",
     "weights/consisgad/elliptic_Naive_ConsisGAD.pt"),
]

THRESHOLDS = [0.3, 0.4, 0.5]

# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_data(cfg_path, weight_path):
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(cfg)
    dataset = dataset.to(device)

    # 推断 input_dim
    if cfg.model.input_dim == "???":
        OmegaConf.update(cfg, "model.input_dim", dataset.x.shape[1])

    model = get_model(cfg, dataset)
    model = model.to(device)

    state = torch.load(weight_path, map_location=device)
    # 有些权重文件直接存 state_dict，有些存 {'model': ...}
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    return model, dataset, cfg


def get_probs_and_labels(model, dataset):
    """返回 (probs_np, labels_np)，只含有标签节点（y != -1）"""
    with torch.no_grad():
        out = model(dataset)
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.sigmoid(out).squeeze().cpu().numpy()

    labels = dataset.y.cpu().numpy()
    valid = labels != -1
    return probs[valid], labels[valid]


def print_distribution(name, probs, labels):
    fraud_probs = probs[labels == 1]
    normal_probs = probs[labels == 0]

    print(f"\n{'='*60}")
    print(f"  模型: {name}")
    print(f"{'='*60}")
    print(f"  有标签节点总数: {len(labels):,}  (欺诈: {labels.sum():,}, 正常: {(labels==0).sum():,})")
    print(f"\n  【欺诈节点 (Class=1) 概率分布】")
    print(f"    min={fraud_probs.min():.4f}  p25={np.percentile(fraud_probs,25):.4f}  "
          f"median={np.median(fraud_probs):.4f}  p75={np.percentile(fraud_probs,75):.4f}  "
          f"max={fraud_probs.max():.4f}")
    print(f"\n  【正常节点 (Class=0) 概率分布】")
    print(f"    min={normal_probs.min():.4f}  p25={np.percentile(normal_probs,25):.4f}  "
          f"median={np.median(normal_probs):.4f}  p75={np.percentile(normal_probs,75):.4f}  "
          f"max={normal_probs.max():.4f}")

    try:
        auc = roc_auc_score(labels, probs)
        print(f"\n  AUC-ROC (阈值无关): {auc:.4f}")
    except Exception:
        pass

    print(f"\n  {'阈值':<8} {'Precision':<12} {'Recall':<10} {'F1':<10} {'#预测为欺诈'}")
    print(f"  {'-'*55}")
    for thr in THRESHOLDS:
        preds = (probs > thr).astype(int)
        p = precision_score(labels, preds, pos_label=1, zero_division=0)
        r = recall_score(labels, preds, pos_label=1, zero_division=0)
        f = f1_score(labels, preds, pos_label=1, zero_division=0)
        n_pred = preds.sum()
        print(f"  {thr:<8.1f} {p:<12.4f} {r:<10.4f} {f:<10.4f} {n_pred:,}")


def main():
    print("\n" + "="*60)
    print("  Threshold 诊断脚本")
    print("  目标：选定 0.3 / 0.4 / 0.5 中最合适的固定评估阈值")
    print("="*60)

    results = {}  # {model_name: {thr: f1}}

    for cfg_path, weight_path in EXPERIMENTS:
        name = os.path.splitext(os.path.basename(cfg_path))[0]

        if not os.path.exists(weight_path):
            print(f"\n[跳过] 权重文件不存在: {weight_path}")
            continue
        if not os.path.exists(cfg_path):
            print(f"\n[跳过] 配置文件不存在: {cfg_path}")
            continue

        try:
            model, dataset, cfg = load_model_and_data(cfg_path, weight_path)
            probs, labels = get_probs_and_labels(model, dataset)
            print_distribution(name, probs, labels)
            results[name] = {}
            for thr in THRESHOLDS:
                preds = (probs > thr).astype(int)
                results[name][thr] = f1_score(labels, preds, pos_label=1, zero_division=0)
        except Exception as e:
            print(f"\n[错误] {name}: {e}")
            import traceback; traceback.print_exc()

    # 汇总表
    if results:
        print(f"\n\n{'='*60}")
        print("  汇总：各模型在不同阈值下的 Binary F1")
        print(f"{'='*60}")
        header = f"  {'模型':<35}"
        for thr in THRESHOLDS:
            header += f"  thr={thr}"
        print(header)
        print(f"  {'-'*55}")
        for name, thr_f1 in results.items():
            row = f"  {name:<35}"
            for thr in THRESHOLDS:
                row += f"  {thr_f1.get(thr, 0):.4f} "
            print(row)

        print("\n  【建议】选 F1 值合理（不过高/不过低）且模型间差距最大的阈值")
        print("          recall ≈ 1.0 表示阈值太低；F1 ≈ 0 表示阈值太高")


if __name__ == "__main__":
    main()
