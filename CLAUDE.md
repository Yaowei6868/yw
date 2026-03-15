# fraud-detection-gnn 项目说明

## 项目简介

本项目是一个基于图神经网络（GNN）的欺诈检测基准框架，支持持续学习（Continual Learning）实验。
数据集涵盖 Elliptic、Elliptic++ Actor、DGraphFin 三个金融图数据集，
支持 GCN、GAT、HOGRL、CGNN、BSL、ConsisGAD、GradGNN、PMP 等多种模型。

---

## 环境要求

- Python 3.8+
- PyTorch（**必须为 CUDA 版本**，详见下方安装说明）
- torch_geometric
- omegaconf、scikit-learn、pandas、numpy、matplotlib、networkx
---

## 安装依赖

### 1. 安装支持 GPU 的 PyTorch（RTX 3090 对应 CUDA 12.1）

```bash
# 卸载已有版本（如有）
pip uninstall torch torchvision torchaudio -y

# 安装 CUDA 12.1 版本（适用于 RTX 3090）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. 安装 PyG 及其扩展

```bash
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu121.html

pip install omegaconf scikit-learn pandas numpy matplotlib networkx tensorboard
```

### 3. 验证 GPU 识别

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# 期望输出: True  /  NVIDIA GeForce RTX 3090
```

---

## 训练

### 单个实验

```bash
python train.py --config configs/<模型目录>/<配置文件>.yaml
```

### 批量运行脚本（位于 scripts/ 目录）

```bash
# 全部三个数据集
nohup bash scripts/run_all_experiments.sh > logs/main.log 2>&1 &

# 仅 Elliptic
nohup bash scripts/run_elliptic_only.sh > logs/elliptic_main.log 2>&1 &

# 仅 Elliptic++ Actor
nohup bash scripts/run_elliptic_actor_only.sh > logs/elliptic_actor_main.log 2>&1 &

# 仅 DGraphFin
nohup bash scripts/run_dgraph_only.sh > logs/dgraph_main.log 2>&1 &
```

> 脚本支持断点续跑：检测到结果 CSV 已完整时自动跳过对应实验。

### 配置文件说明

| 字段 | 说明 |
|---|---|
| `train.device` | 设置为 `cuda` 使用 GPU，`cpu` 使用 CPU |
| `train.model` | 模型名称，如 `consisgad`、`bsl`、`hogrl`、`cgnn`、`grad`、`gcn`、`gat` 等 |
| `train.dataset` | 数据集名称：`elliptic`、`elliptic_actor`、`dgraphfin` |
| `train.num_epochs_per_task` | 每个时间任务的训练轮数 |
| `train.task_schedule` | 时间步划分列表，每项为 `[start, end]` |
| `train.ewc_lambda` | EWC 正则化强度，0 表示关闭 |
| `train.lwf_alpha` | LwF 蒸馏损失权重，0 表示关闭 |
| `train.buffer_size_per_class` | Experience Replay 每类缓冲区大小，0 表示关闭 |

---

## 数据目录结构

```
data/
  elliptic/
    elliptic_txs_features.csv
    elliptic_txs_edgelist.csv
    elliptic_txs_classes.csv
  elliptic++actor/
    (InMemoryDataset 自动处理)
  dgraphfin/
    raw/DGraphFin.zip  (需从 https://dgraph.xinye.com 手动下载)
    processed/         (首次运行后自动生成)
```

---

## 输出

- 模型权重保存于 `weights/<模型名>/` 目录（由 yaml 中 `train.save_dir` 指定）
- 训练指标 CSV 保存于 `weights/<模型名>/metrics/`
- TensorBoard 日志保存于 `runs/<config.name>/`

查看 TensorBoard：

```bash
tensorboard --logdir runs/
```

### 结果汇总工具（位于 tools/ 目录）

```bash
python tools/collect_results.py    # 汇总所有实验指标到 results_summary.csv
python tools/analyze_results.py    # 分析和可视化结果
```

---

## 支持的持续学习策略

| 策略 | yaml 配置方式 |
|---|---|
| Naive（无策略） | `ewc_lambda: 0`, `lwf_alpha: 0`, `buffer_size_per_class: 0` |
| EWC | `ewc_lambda: 1.0`（调大则约束越强） |
| LwF | `lwf_alpha: 1.0` |
| Experience Replay | `buffer_size_per_class: 50`（每类保留 50 个样本） |

---

## ⚠️ 未完成实现的模型（暂不可运行）

| 模型 | 状态 | 原因 |
|---|---|---|
| EvolveGCN | ❌ 空壳占位符 | 依赖时序 GRU/LSTM 更新图参数，实现复杂，暂未完成 |
| TGN | ❌ 空壳占位符 | 依赖时序记忆模块和时间编码，实现复杂，暂未完成 |

对应 configs 位于 `configs/dynamic_graph/`，但**不要在脚本中包含这两个实验**，运行会直接报错。

---

## 已验证可运行的实验

| 模型 | 数据集 | 策略 | 权重路径 |
|---|---|---|---|
| BSL | elliptic | Naive | `weights/bsl/` |
| GradGNN | elliptic | Naive | `weights/grad/` |
| HOGRL | elliptic | Naive | `weights/hogrl/` |
| CGNN | elliptic | Naive | `weights/cgnn/` |
| ConsisGAD | elliptic | Naive | `weights/consisgad/` |
| ConsisGAD | elliptic | CL | `weights/consisgad/` |
