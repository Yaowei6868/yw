# fraud-detection-gnn 项目说明

## 项目简介

本项目是一个基于图神经网络（GNN）的欺诈检测基准框架，支持持续学习（Continual Learning）实验。
数据集以比特币交易图（Elliptic、Elliptic++ Actor）和金融图（T-Finance）为主，
涵盖 GCN、GAT、HOGRL、CGNN、BSL、ConsisGAD、GradGNN、PMP 等多种模型。

---

## 环境要求

- Python 3.8+
- PyTorch（**必须为 CUDA 版本**，详见下方安装说明）
- torch_geometric
- omegaconf、scikit-learn、pandas、numpy、matplotlib、networkx

---

## 安装依赖

### 1. 安装支持 GPU 的 PyTorch（RTX 4090 对应 CUDA 12.x）

```bash
# 卸载 CPU 版本（如果已安装）
pip uninstall torch torchvision torchaudio -y

# 安装 CUDA 12.1 版本（适用于 RTX 4090）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> **注意**：当前环境安装的是 `torch 2.9.0+cpu`（纯 CPU 版），RTX 4090 无法被识别。
> 必须重新安装 CUDA 版本才能启用 GPU 训练。

### 2. 安装 PyG 及其他依赖

```bash
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.x.0+cu121.html

pip install omegaconf scikit-learn pandas numpy matplotlib networkx tensorboard
```

### 3. 验证 GPU 识别

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# 期望输出: True  /  NVIDIA GeForce RTX 4090
```

---

## 训练

### 基本命令

```bash
python train.py --config configs/<模型目录>/<配置文件>.yaml
```

### 示例

```bash
# ConsisGAD (Naive，无持续学习策略)
python train.py --config configs/ConsisGAD/elliptic_Naive_ConsisGAD.yaml

# BSL (Naive)
python train.py --config configs/BSL/elliptic_Naive_BSL.yaml

# HOGRL (Naive)
python train.py --config configs/HOGRL/elliptic_Naive_HOGRL.yaml

# CGNN (Naive)
python train.py --config configs/CGNN/elliptic_Naive_CGNN.yaml

# GradGNN (Naive)
python train.py --config configs/Grad/elliptic_Naive_Grad.yaml

# GAT (CL，使用 EWC 策略)
python train.py --config configs/traditional/GAT/elliptic_CL_GAT.yaml
```

### 配置文件说明

| 字段 | 说明 |
|---|---|
| `train.device` | 设置为 `cuda` 使用 GPU，`cpu` 使用 CPU |
| `train.model` | 模型名称，如 `consisgad`、`bsl`、`hogrl`、`cgnn`、`grad`、`gcn`、`gat` 等 |
| `train.dataset` | 数据集名称：`elliptic`、`elliptic_actor`、`tfinance` |
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
  tfinance/
    raw/tfinance   (DGL 格式原始文件，需手动下载)
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

---

## 支持的持续学习策略

| 策略 | yaml 配置方式 |
|---|---|
| Naive（无策略） | `ewc_lambda: 0`, `lwf_alpha: 0`, `buffer_size_per_class: 0` |
| EWC | `ewc_lambda: 1.0`（调大则约束越强） |
| LwF | `lwf_alpha: 1.0` |
| Experience Replay | `buffer_size_per_class: 50`（每类保留 50 个样本） |

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
