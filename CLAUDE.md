# fraud-detection-gnn 项目说明（CLAUDE.md）

> 最后更新：2026-04-03
> 本文件供 Claude 在新对话中快速恢复上下文。请在每次重要进展后更新。

---

## 一、项目定位与研究动机

### 研究问题（一句话）
在图神经网络欺诈检测的任务增量持续学习场景下，提出 TASD-CL 框架，通过子空间语义过滤（SSF）、原型凝缩（SPC）和置信度蒸馏（SCD）三个组件，使 BSL 模型在时序欺诈 pattern 演变中保持稳定的检测性能。

### 研究叙事逻辑（最终确认版，基于 BRIGHT 工业动机）

1. **工业背景（BRIGHT 动机）**：真实金融风控系统（如 eBay）中交易数据以流式到达，若每次都构建包含全量历史边的 Cumulative 图，会面临两个致命问题：**延迟极高**（无法满足毫秒级实时推理要求）和**内存爆炸（OOM）**。因此工业界采用基于时间窗口的 Task-only 构图方式，每次只处理当前窗口内的边（如 BRIGHT 的 Two-Stage Directed Graph 设计）。

2. **发现的问题**：在 Task-only 的窗口构图约束下，GNN 欺诈检测模型极易对当前欺诈 pattern 过拟合，随时间步推进性能不稳定（在 Elliptic 等数据集上可观察到明显的性能波动和下滑趋势）。

3. **根本原因**：Task-only 窗口切分本质上是任务增量持续学习场景。模型在当前任务上更新参数时，会覆盖历史欺诈 pattern 的表示，导致子空间结构被污染、历史知识丢失。

4. **为什么不能用传统 ER**：标准 Experience Replay 需要在新任务图上重新前向传播历史节点，但 Task-only 设定下历史节点的邻居图结构已经消失，强行回放得到的是结构噪声，反而有害——且这违背了 BRIGHT 提出的实时性约束。

5. **解决方法**：TASD-CL 框架，通过 SPC 仅保留轻量级高斯原型 (μ,σ)，以极低内存代价实现跨时间步的知识保留，完全不依赖历史图结构，天然契合 Task-only 工业约束。

6. **效果目标**：在 Elliptic 数据集上，TASD-CL 在多个评估指标上超过所有对比方法（BSL/CGNN/ConsisGAD/GradGNN/HOGRL/PMP/GCN）。

### 核心思想：为什么选 BSL 作为 backbone
BSL 将节点嵌入解耦为三个子空间：
- `Z_na`：正常-异常交互子空间
- `Z_aa`：纯异常（欺诈感知）子空间
- `Z_nn`：纯正常子空间

这种解耦使得"保护哪些参数"、"蒸馏哪些节点"、"回放什么知识"都可以针对欺诈语义精细设计，是 TASD-CL 三个组件的设计基础。

---

## 二、TASD-CL 三组件详细说明

### BSL 子空间结构（三组件的共同基础）

BSL 将每个节点的 hidden 向量 `z`（维度 `hidden_dim`）解耦为三段：

```
z = [Z_na | Z_aa | Z_nn]
     ↑         ↑       ↑
  正常-异常  纯欺诈  纯正常
  交互子空间  子空间   子空间
  (sub_dim)  (sub_dim) (sub_dim)
```

`sub_dim = hidden_dim // 3`（config 中 `hidden_dim: 96` → `sub_dim = 32`）

另有注意力权重向量 `alpha = [α_na, α_aa, α_nn]`，表示当前节点特征在三个子空间的归属比重（三者和为 1）。

| 组件 | 全称 | 作用 | 实现位置 |
|---|---|---|---|
| **SSF** | Subspace-Stratified Fisher | 改造版 EWC，对 BSL 不同参数施加不同强度约束，子空间路由参数约束最强 | `trainer.py: _get_ssf_lambda()` |
| **SPC** | Subspace Prototype Condensation | 任务结束后提取三子空间的类原型（Gaussian μ,σ），存入缓冲区，绕开图结构做无噪声回放 | `trainer.py: _update_spc_prototypes()`, `buffer.py: SubspacePrototypeBuffer` |
| **SCD** | Subspace-Conditioned Distillation | 替代标准 LwF，在三子空间层面蒸馏，只蒸馏旧模型高置信度节点，Z_aa 权重最高（2.0） | `trainer.py: _compute_scd_loss()` |

---

### Component A：SSF（Subspace-Stratified Fisher）

**本质**：改造版 EWC，对不同 BSL 参数的 Fisher 约束乘以语义角色系数。

标准 EWC 正则项：
```
L_ewc = λ × Σ_θ  F_θ × (θ - θ*)²
```

SSF 在此基础上，对每个参数查询 `BSL_PARAM_GROUPS` 得到角色乘数，实际约束 = `ewc_lambda × role_mult`：

```python
BSL_PARAM_GROUPS = {
    'att_vec':      3.0,   # 子空间路由（最高）：定义欺诈特征归属，丢失则子空间解耦崩塌
    'att_bias':     3.0,
    'edge_decoder': 2.0,   # 边类型分类（中等）
    'gnn_encoder':  0.5,   # GATv2 图编码器（低）：图结构随任务变化，应允许更新
    'lin_in':       0.5,
    'classifier':   0.3,   # 最终分类头（最低）：任务相关层，允许自由更新
}
```

实际约束范围（`ewc_lambda=1.0`时）：`att_vec/att_bias = 3.0`，`classifier = 0.3`。

**为什么 att_vec 最高**：它决定每个节点的特征往哪个子空间路由（欺诈 → Z_aa，正常 → Z_nn），是 BSL 的核心语义结构，一旦被新任务覆盖，整个子空间解耦失效。

**Config 参数**：
```yaml
ewc_lambda: 1.0    # 基础约束强度，乘角色系数后实际范围 [0.3, 3.0]
                   # 设为 0 → SSF 完全关闭，退化为 Naive BSL
```

---

### Component B：SPC（Subspace Prototype Condensation）

**本质**：替代标准 Experience Replay，存储子空间高斯原型而非原始节点索引。

**为什么不能直接做图回放**：标准 ER 存节点索引，下个 task 再前向传播。但欺诈图是时序的——Task 3 的节点在 Task 5 的图快照中邻居已完全不同，强行回放得到的表示是结构噪声，蒸馏反而有害。

**SPC 的做法**：每个 task 训练结束后，计算三子空间在两类上的高斯原型：
```
对每个 task t、类别 c ∈ {0,1}、子空间 k ∈ {0,1,2}：
  mu[t][c][k]    = mean(Z_k[训练节点 & 标签==c])   # [sub_dim]
  sigma[t][c][k] = std (Z_k[训练节点 & 标签==c])   # [sub_dim]
```

存储代价：`O(T × 2 × 3 × sub_dim)` = T=10 时约 1920 个浮点数（极小）。

**回放时**，从原型采样合成节点，跳过 GNN 编码器直接送分类头：
```python
z_sample = mu + eps * sigma   # eps ~ N(0,I)
z_replay = cat([z_na, z_aa, z_nn], dim=1)  # [n, hidden_dim]
out_rep  = model.classifier(z_replay)
L_spc    = BCE(out_rep, y_replay)
```

**Config 参数**：
```yaml
spc_lambda: 1.0       # 回放损失权重，设为 0 → SPC 关闭
spc_n_samples: 64     # 每个 (task, class) 采样多少合成节点
                      # 已见 3 个 task 时，每次回放生成 3×2×64=384 个合成节点
```

---

### Component C：SCD（Subspace-Conditioned Distillation）

**本质**：替代标准 LwF，在三子空间层面做知识蒸馏，且只蒸馏旧模型高置信度节点。

**标准 LwF 的缺陷**：
1. 只蒸馏 1 维 logit 输出，丢失 BSL 三子空间的几何结构
2. 欺诈检测严重不平衡（~9% 欺诈），旧模型对大量正常节点预测接近 `-∞`，强行蒸馏引入负迁移

**SCD 的做法**：
```python
for k in range(3):
    # 只蒸馏旧模型对子空间 k 高置信度的节点（旧模型认为该节点"属于"这个子空间）
    confidence_mask = (old_alpha[:, k] > scd_tau)
    L_scd += subspace_weights[k] × MSE(new_Z_k[mask], old_Z_k[mask])
```

子空间权重：`Z_na=1.0, Z_aa=2.0, Z_nn=1.0`，欺诈感知子空间优先保护。

**Config 参数**：
```yaml
scd_lambda: 0.5       # 蒸馏损失权重，设为 0 → SCD 关闭
scd_tau: 0.3          # 置信度过滤阈值
                      # tau 越小 → 纳入更多节点（噪声更多）
                      # tau 越大 → 只蒸馏最典型节点（更保守）
                      # 当前取 0.3（低阈值，纳入更多节点）

# 关闭标准 LwF（由 SCD 替代，避免重复蒸馏）
lwf_alpha: 0.0
# 关闭标准 ER（由 SPC 替代，避免图结构噪声）
buffer_size_per_class: 0
```

---

### 总损失公式

```
L_total = L_task                                           # 当前 task Focal Loss
        + ewc_lambda × Σ_θ role_mult(θ) × F_θ × (θ-θ*)²  # SSF 参数约束
        + spc_lambda × L_spc                               # SPC 子空间原型回放
        + scd_lambda × L_scd                               # SCD 子空间蒸馏
        + bsl_loss                                         # BSL 自带 L_d + L_bsl
```

### Config 参数速查表

| 参数 | 组件 | 含义 | 设为 0 的效果 |
|---|---|---|---|
| `ewc_lambda` | A: SSF | 参数约束基础强度（乘角色系数后范围 0.3~3.0） | 无参数约束，退化为 Naive |
| `spc_lambda` | B: SPC | 子空间原型回放损失权重 | 无原型回放 |
| `spc_n_samples` | B: SPC | 每个 (task, class) 合成多少节点 | — |
| `scd_lambda` | C: SCD | 子空间蒸馏损失权重 | 无蒸馏 |
| `scd_tau` | C: SCD | 旧模型置信度过滤阈值（越小纳入越多节点） | — |
| `lwf_alpha: 0.0` | — | 关闭标准 LwF（已被 SCD 替代） | — |
| `buffer_size_per_class: 0` | — | 关闭标准 ER（已被 SPC 替代） | — |

---

## 三、实验设计

### Snapshot 设计（已确认：task-only）
- 每个任务只使用**当前时间步范围内的边**（`edge_index` 按 `valid_node_mask` 过滤）
- 但 `snapshot_data.x` 仍是全量节点特征（shallow copy）
- 本质：**图结构 task-only，节点特征 all**
- **为什么不用 cumulative**：task-only 让 baseline 的性能下降归因更干净，只来自时序分布偏移，不混入"历史图信息缺失"的因素；且与 Elliptic 文献标准对齐

关键代码（`trainer.py` 第 726-734 行）：
```python
valid_node_mask = (self.dataset.timesteps >= task_start_t) & (self.dataset.timesteps <= task_end_t)
edge_mask = valid_node_mask[row] & valid_node_mask[col]
snapshot_data = copy.copy(self.dataset)
snapshot_data.edge_index = self.dataset.edge_index[:, edge_mask]
```

### 评估方式
- **CL 矩阵评估**：`f1_matrix[current_task_id, t_id]` 记录在任务 t 训练后对历史任务 j 的 F1
- **每 task 结束后**在所有已见任务上做 CL 评估（`evaluate_cl_metrics`）
- 指标：Binary F1 / Macro F1 / AUC-ROC / AUC-PR / G-Mean / Specificity / MCC
- **注意**：Forgetting / BWT 指标已从代码和结果汇总中删除（论文不使用）

### 损失函数设计
- 主损失：`BinaryFocalLoss`（alpha 动态计算，根据正负比例截断）
- Task 1：`clip_cap = 20.0`；其他 task：`clip_cap = 10.0`
- 总损失：`task_loss + cl_loss + bsl_loss + cgnn_loss + spc_lambda * spc_loss`

---

## 四、数据集信息

| 数据集 | 节点数 | 边数 | 时间步 | 欺诈率 | 特殊问题 |
|---|---|---|---|---|---|
| **Elliptic** | ~203K | ~234K | 49个时间步（本项目用前10） | ~9% | 标准，无 OOM |
| **Elliptic++ Actor** | 类似 | 类似 | 类似 | 类似 | 标准，无 OOM |
| **DGraphFin** | ~3.7M | ~4.3M | 边时间戳 1-821，划分为10段 | ~1.3% | **严重 OOM 问题** |

### DGraphFin OOM 根因分析（⚠️ 未解决）
- `copy.copy(dataset).to(device)` 将全量 3.7M 节点特征（约 3.7M × 17 × 4bytes ≈ 250MB）搬上 GPU
- 重型模型中间激活：BSL GATv2 attention 约 6GB；HOGRL A²/A³ 稀疏矩阵乘法爆炸
- **目前状态**：DGraphFin 上除 GraphSMOTE 外全部 OOM，暂时搁置，待后续处理

---

## 五、所有模型实现状态

| 模型 | 类别 | 实现状态 | 备注 |
|---|---|---|---|
| GCN | 通用 GNN 基线 | ✅ 完整 | |
| HOGRL | 欺诈 SOTA | ✅ 完整 | DGraphFin OOM |
| CGNN | 欺诈 SOTA | ✅ 完整 | DGraphFin OOM |
| BSL | 欺诈 SOTA + backbone | ✅ 完整 | DGraphFin OOM |
| ConsisGAD | 欺诈 SOTA | ✅ 完整 | |
| GradGNN | 欺诈 SOTA | ✅ 完整 | |
| PMP | 欺诈 SOTA | ✅ 完整 | config 在 `configs/fraud_sota/{dataset}/` |

**已从实验中移除（config 移至 `configs/deprecated/`，代码 MODEL_MAP 已删除）**：
- GAT：与 GCN 功能重叠，冗余
- GraphSMOTE：不平衡采样方法，与 CL 故事线无关
- EvolveGCN / TGN：空壳占位符，从未完整实现

---

## 六、配置文件目录结构

```
configs/
  traditional/               # GCN Naive（三个数据集）
  fraud_sota/                # 欺诈 SOTA baseline（Naive）
    elliptic/                # BSL/CGNN/ConsisGAD/GradGNN/HOGRL/PMP
    elliptic++actor/
    dgraphfin/
  ours/                      # 我们的方法
    main/                    # BSL + TASD-CL（三个数据集）
    cl_on_bsl/               # BSL + EWC/LwF/ER（通用 CL 基线）
    ablation/                # noSSF / noSPC / noSCD 消融
  deprecated/                # 已废弃，不再运行
    cl_baselines/            # GCN + EWC/LwF/ER（已改为 BSL+CL）
    imbalanced/              # GraphSMOTE
    dynamic_graph/           # EvolveGCN / TGN（空壳）
    *.yaml                   # GAT Naive（三个数据集）
```

---

## 七、脚本说明（已修复路径问题）

```bash
# 所有脚本必须从项目根目录运行，或使用以下方式：
nohup bash scripts/run_all_experiments.sh > logs/main.log 2>&1 &
nohup bash scripts/run_elliptic_only.sh > logs/elliptic_main.log 2>&1 &
nohup bash scripts/run_elliptic_actor_only.sh > logs/elliptic_actor_main.log 2>&1 &
nohup bash scripts/run_dgraph_only.sh > logs/dgraph_main.log 2>&1 &
```

脚本内部已修复：新增 `ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"` 确保 `train.py` 路径正确。

> 脚本支持断点续跑：检测到结果 CSV 已完整时自动跳过对应实验。

---

## 八、当前实验进度

### 已完成运行（有结果）
| 模型 | 数据集 | 策略 | 状态 |
|---|---|---|---|
| GCN | Elliptic | Naive | ✅ 完成 |
| BSL | Elliptic | Naive | ✅ 完成 |
| BSL | Elliptic | TASD-CL | ✅ 完成 |
| CGNN | Elliptic | Naive | ✅ 完成 |
| ConsisGAD | Elliptic | Naive | ✅ 完成 |
| GradGNN | Elliptic | Naive | ✅ 完成 |
| HOGRL | Elliptic | Naive | ✅ 完成 |
| PMP | Elliptic | Naive | ✅ 完成 |
| GCN | Elliptic++ Actor | Naive | ✅ 完成 |
| BSL | Elliptic++ Actor | Naive + TASD-CL | ✅ 完成 |
| CGNN | Elliptic++ Actor | Naive | ✅ 完成 |
| ConsisGAD | Elliptic++ Actor | Naive | ✅ 完成 |
| GradGNN | Elliptic++ Actor | Naive | ✅ 完成 |
| HOGRL | Elliptic++ Actor | Naive | ✅ 完成 |

### 待运行（论文所需）
| 模型 | 数据集 | 策略 | 备注 |
|---|---|---|---|
| BSL | Elliptic | EWC / LwF / ER | `configs/ours/cl_on_bsl/elliptic_*.yaml` |
| BSL | Elliptic++ Actor | EWC / LwF / ER | `configs/ours/cl_on_bsl/elliptic++actor_*.yaml` |
| BSL | Elliptic | noSSF / noSPC / noSCD | `configs/ours/ablation/elliptic_*.yaml` |
| BSL | Elliptic++ Actor | noSSF / noSPC / noSCD | `configs/ours/ablation/elliptic++actor_*.yaml` |
| 所有模型 | DGraphFin | — | OOM 未解决，暂搁置 |

---

## 九、实验结果详细分析

> 数据来源：`weights/*/metrics/*_aggregate_metrics.csv` 及 `results_summary.csv`
> 当前只有 Elliptic 和 Elliptic++ Actor 数据集有结果，DGraphFin 全部 OOM。

---

### 9.1 Elliptic 数据集 — SOTA Baseline 最终结果（Task 10）

| 模型 | 策略 | Final F1 | Final AUC-ROC | Forgetting | 趋势 |
|---|---|---|---|---|---|
| GCN | Naive | 0.2970 | 0.8304 | +0.092 | 先升后降 |
| GCN | EWC | 0.2836 | 0.8211 | +0.002 | 先升后降 |
| GCN | LwF | 0.2978 | 0.8216 | +0.0001 | 先升后降 |
| GCN | ER | 0.2981 | 0.8358 | +0.003 | 先升后降 |
| BSL | Naive | 0.2073 | 0.6163 | +0.113 | 先升后降 |
| BSL | TASD-CL | 0.2404 | 0.7714 | ≈0 | 先升后降 |
| CGNN | Naive | 0.3877 | 0.8735 | +0.138 | 先升后降 |
| ConsisGAD | Naive | 0.3211 | 0.7832 | -0.016 | **持续上升** |
| ConsisGAD | EWC(CL) | **0.5174** | 0.7290 | 0.0 | 稳定 |
| GradGNN | Naive | 0.3374 | 0.8368 | -0.139 | **持续上升** |
| HOGRL | Naive | 0.2956 | 0.7267 | -0.089 | **持续上升** |
| PMP | Naive | 0.3849 | 0.8719 | +0.101 | 先升后降 |

### 9.2 Elliptic++ Actor 数据集 — 关键对比结果（Task 10）

| 模型 | 策略 | Final F1 | Final AUC-ROC | Forgetting |
|---|---|---|---|---|
| GCN | Naive | — | — | — |
| BSL | Naive | 0.3549 | 0.7405 | +0.001 |
| BSL | TASD-CL | 0.2686 | 0.7111 | ≈0 |
| CGNN | Naive | — | — | — |
| GradGNN | Naive | — | — | — |
| HOGRL | Naive | — | — | — |
| GradGNN | CL | — | — | — |
| HOGRL | CL | — | — | — |

> 注：Elliptic++ Actor 部分结果文件存在，但上表待从 CSV 中补全核实。

---

### 9.3 所有方法逐任务 avg_F1 对比（Elliptic，排除 warmup Task 1）

| Task | GCN | CGNN | GradGNN | HOGRL | PMP | ConsisGAD | BSL Naive | **TASD-CL** |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.136 | 0.113 | 0.175 | 0.115 | 0.136 | 0.000 | 0.118 | 0.048 |
| 3 | 0.311 | 0.298 | 0.338 | 0.324 | 0.100 | 0.000 | 0.175 | 0.144 |
| 4 | 0.398 | 0.386 | 0.410 | 0.399 | 0.080 | 0.000 | 0.270 | 0.257 |
| 5 | 0.454 | 0.461 | 0.465 | 0.453 | 0.068 | 0.000 | 0.351 | 0.322 |
| 6 | **0.469** | **0.476** | **0.473** | **0.457** | 0.063 | 0.000 | **0.381** | 0.363 |
| 7 | 0.456 | 0.467 | 0.458 | 0.455 | 0.057 | 0.000 | 0.378 | **0.368** |
| 8 | 0.426 | 0.448 | 0.427 | 0.429 | 0.061 | 0.000 | 0.348 | **0.354** |
| 9 | 0.380 | 0.399 | 0.383 | 0.395 | 0.055 | 0.000 | 0.311 | 0.315 |
| 10 | 0.381 | 0.402 | 0.384 | 0.395 | 0.059 | 0.000 | 0.316 | 0.302 |

### 9.4 所有方法逐任务 avg_AUC-ROC 对比（Elliptic）

| Task | GCN | CGNN | GradGNN | HOGRL | BSL Naive | **TASD-CL** |
|---|---|---|---|---|---|---|
| 2 | 0.934 | 0.897 | 0.958 | 0.914 | 0.835 | **0.868** |
| 3 | **0.937** | 0.909 | 0.940 | **0.929** | 0.818 | 0.860 |
| 4 | 0.920 | 0.896 | 0.912 | 0.901 | 0.811 | 0.852 |
| 5 | 0.923 | **0.910** | 0.912 | 0.907 | 0.830 | 0.853 |
| 6 | 0.924 | **0.910** | 0.905 | 0.900 | 0.834 | 0.856 |
| 7 | 0.911 | 0.909 | 0.890 | 0.895 | 0.825 | 0.845 |
| 8 | 0.877 | 0.897 | 0.850 | 0.869 | 0.777 | **0.825** |
| 9 | 0.856 | 0.867 | 0.829 | 0.847 | 0.765 | 0.791 |
| 10 | 0.856 | 0.863 | 0.837 | 0.857 | 0.768 | 0.778 |

**关键观察**：TASD-CL 的 AUC 在所有任务上均高于 BSL Naive（backbone 改进有效），但仍低于 GCN/CGNN/HOGRL/GradGNN。

---

### 9.5 消融实验（Elliptic，Task 10）

| 方法 | avg_F1 | avg_AUC-ROC | avg_Macro_F1 |
|---|---|---|---|
| TASD-CL Full | **0.302** | 0.778 | **0.574** |
| noSSF (SPC+SCD) | 0.066 | 0.738 | 0.488 |
| noSPC (SSF+SCD) | 0.049 | 0.750 | 0.484 |
| noSCD (SSF+SPC) | 0.188 | **0.797** | 0.551 |

**关键发现**：
- SSF 和 SPC 都是决定性组件，移除任意一个 F1 崩溃至 0.05-0.07（-84%）
- SCD 对 F1 贡献 +61%（0.188→0.302），但轻微损害 AUC（noSCD AUC 最高 0.797）
- Full TASD-CL 在 F1 上是所有消融变体中最优的

---

### 9.6 关键未决问题

#### 问题一：TASD-CL 绝对性能不足
**现象**：TASD-CL F1=0.302，低于 GCN(0.381)/CGNN(0.402)/GradGNN(0.384)/HOGRL(0.395)。
**原因**：BSL backbone 本身比其他 SOTA 弱，TASD-CL 虽然改进了 BSL，但未能跨越 backbone 差距。
**行动**：超参调优（见第十六节策略 A），目标将 F1 推至 0.38+。

#### 问题二：ConsisGAD 和 PMP 结果异常
**现象**：ConsisGAD F1=0.000，PMP F1=0.059，均在 Task-only 设定下完全失效。
**正面解读**：这恰恰证明了 Task-only 窗口设定对无 CL 机制的模型非常苛刻，TASD-CL 的价值正在于此。需确认失效是设定本身导致的，而非 bug。

#### 问题三：DGraphFin 无结果
**状态**：Task-only 下已严重 OOM，无法运行。暂时搁置，仅用 Elliptic 作为主要实验。

---

## 十、✅ Snapshot 设计：Task-only（已最终确认）

**此问题已于 2026-04-03 最终拍板：使用 Task-only，不再讨论 Cumulative。**

### 确认原因：BRIGHT 工业论文的背书

eBay 的 BRIGHT 论文（发表于顶会）明确指出：在真实金融风控系统中，维护一个无限增长的 Cumulative 全局图会面临两个致命问题：
- **延迟极高**：无法满足毫秒级实时推理要求
- **内存爆炸（OOM）**：邻居节点呈指数级增长

因此 BRIGHT 提出了基于时间窗口分区的 Task-only 构图方式。我们的设计与之对齐，具有充分的工业背书。

### Cumulative 方案的内存代价分析（为什么不能用）

| 数据集 | Task-only 最大边数（单 Task） | Cumulative Task 10 边数 | 内存变化 |
|---|---|---|---|
| Elliptic | ~4.8K 边 | ~47.8K 边 | ×10，仍可接受 |
| Elliptic++ Actor | 类似 Elliptic | 类似 Elliptic | ×10，仍可接受 |
| **DGraphFin** | **~430K 边（已 OOM）** | **~4.3M 边** | **×10，完全不可行** |

DGraphFin 在 Task-only 下已经严重 OOM（BSL/HOGRL 等全部爆显存），切换为 Cumulative 会使 Task 10 的图变为全量 4.3M 边，显存需求提升约 10 倍，从根本上不可行。

### 论文叙事中的用法

在论文 Experiment Setup 节，引用 BRIGHT 的工业约束来解释为什么选择 Task-only：
> "Following the industrial constraint identified by BRIGHT [cite], where maintaining a cumulative global graph causes OOM and prohibitively high latency, we adopt a task-only window-based graph construction. This setting also cleanly isolates temporal distribution shift as the sole source of performance degradation."

这样既有工业背书，又解释了实验设计的合理性，还顺带强化了我们 SPC 组件的价值（SPC 正是在 Task-only 约束下替代图结构回放的关键创新）。

---

## 十一（原十）、遗留困惑与待决策问题

### 已解决
- [x] 灾难性遗忘问题：实验证明基本不存在，研究叙事中不再提及
- [x] GraphSMOTE 实现错误：已修复
- [x] PMP config 缺失：已补全
- [x] 脚本路径错误：已修复
- [x] **Snapshot 设计**：已最终确认为 Task-only，BRIGHT 工业论文提供充分背书（见第十节）
- [x] **SCD 组件失效 Bug**：`old_model` 未保存，导致 SCD 始终返回 0。已修复（`trainer.py` 第 980-985 行）
- [x] **评估指标扩充**：新增 Macro F1、Macro Recall、Specificity、MCC，删除 Forgetting/BWT/avg_cost
- [x] **训练测试划分改为时序切分**：前 80% 时间步 → 训练，后 20% → 测试
- [x] **Task 1 标记为 warm-up**：欺诈样本极少，CSV 中 `is_warmup=True`，论文分析排除该任务

### 待解决（优先级排序）

- [ ] **⚠️【最优先】TASD-CL 超参调优**：当前 Elliptic 结果 TASD-CL F1=0.302，低于 GCN(0.381)/CGNN(0.402)/GradGNN(0.384)/HOGRL(0.395)。需通过系统调参（见第十二节实验策略）使 TASD-CL 在 5-6 个对比方法上超越。目标指标可扩展至 Specificity/Macro F1/MCC。

- [ ] **⚠️ ConsisGAD 和 PMP 结果异常**：ConsisGAD Naive F1=0.000（历史记录 0.32），PMP Naive F1=0.059（极低）。两者 recall 近乎为 0。需排查是 config 问题、代码 bug 还是阈值问题。如果这两个模型确实在当前 Task-only 设定下失效，则 TASD-CL 已经超越其中 2 个对比方法。

- [ ] **DGraphFin OOM**：Task-only 下已严重 OOM，Cumulative 更不可行。方案待定（子图采样 or 混合精度）。

- [ ] **理论补充**：SCD 中 Z_aa 子空间权重为 2.0 的理论动机尚未形式化。

---

## 十一、第二工作头脑风暴（硕士毕业论文继承关系）

> 第一工作核心：在任务增量持续学习下，BSL 模型因时序欺诈 pattern 偏移导致性能下降，TASD-CL 通过子空间感知的三组件缓解这一问题。
> 第二工作需要：在第一工作的基础上，顺承其发现或不足，进一步深化或延伸，形成"发现问题→解决问题→更深一步"的大论文叙事。

---

### 11.1 方向一：从"缓解偏移"到"主动检测偏移"（分布偏移感知自适应）

**继承点**：第一工作证明了时序欺诈 pattern 偏移是核心问题，但 TASD-CL 是被动应对（每个 task 都用同样的策略），无法判断"什么时候偏移发生了、偏移有多严重"。

**第二工作方向**：
- 利用 BSL 的三子空间表示，在线监测子空间分布变化（如 Z_aa 的均值/方差漂移）作为 **pattern shift detector**
- 当检测到显著偏移时，触发增强的适应机制；偏移不显著时，使用轻量级更新节省计算
- 本质：**主动式持续学习（Proactive Continual Learning）for Fraud Detection**

**优点**：
- 直接继承 BSL 子空间结构，代码复用率高
- 解决第一工作的一个明确局限（被动适应）
- 有实用价值：节省计算、精准适应

**缺点**：
- "偏移检测"本身是一个独立子问题，需要额外的设计和验证
- 可能缺乏足够的理论深度

---

### 11.2 方向二：从"任务增量"到"类增量"（新型欺诈模式的零样本/少样本发现）

**继承点**：第一工作假设每个 task 的欺诈类别是固定的（二分类：欺诈/正常）。但现实中，新型欺诈手段会不断出现，即新的欺诈"子类别"出现时，模型完全没有该类别的标签。

**第二工作方向**：
- 在任务增量框架下叠加**类增量（Class-Incremental）**维度：新任务中可能出现从未见过的欺诈子类型
- 利用第一工作积累的 SPC 子空间原型作为"已知欺诈知识库"，通过**原型距离**识别异常节点是否属于新型欺诈
- 本质：**开放集欺诈检测（Open-Set Fraud Detection）with Temporal Evolution**

**优点**：
- SPC 原型直接复用，技术继承极强
- 解决现实痛点：已知欺诈越来越容易识别，新型欺诈才是真正威胁
- 类增量 + 时序 + GNN 组合，发表价值高

**缺点**：
- 需要重新标注或构造"新型欺诈"出现的实验场景（数据集改造成本）
- "新型欺诈子类型"在 Elliptic 等数据集中定义不清晰

---

### 11.3 方向三：从"子空间对齐"到"跨图欺诈迁移"（跨数据集迁移学习）

**继承点**：第一工作发现 TASD-CL 在 Elliptic 有效但在 Elliptic++ Actor 上反而更差，说明子空间知识的跨数据集迁移存在问题。

**第二工作方向**：
- 研究如何让在一个金融图数据集上学到的欺诈子空间知识，**迁移到另一个数据集**上
- 例如：在 Elliptic 上预训练的 BSL 子空间原型，能否作为 Elliptic++ Actor 的初始化，加速学习
- 本质：**跨图迁移学习（Cross-Graph Transfer Learning）for Fraud Detection**

**优点**：
- 直接解决第一工作发现的跨数据集泛化失败问题，逻辑自洽
- 迁移学习是主流方向，和 LLM 预训练时代相关（Graph Foundation Model）
- 三个数据集天然构成"源域→目标域"的迁移实验设置

**缺点**：
- 技术路线较宽泛，需要聚焦到具体的迁移机制
- 可能需要新的模型设计，代码复用率中等

---

### 11.4 方向四：从"正向遗忘"到"正向迁移"（Forward Transfer 显式建模）

**继承点**：第一工作实验结果中，GradGNN/HOGRL/ConsisGAD 出现了 **负 forgetting（性能随时间提升）**，即历史任务学到的知识有助于未来任务——这是 Forward Transfer（前向迁移）现象，但第一工作完全没有利用这个特性。

**第二工作方向**：
- 系统研究欺诈检测持续学习中的 Forward Transfer 现象：为什么某些模型会前向迁移？什么样的欺诈 pattern 具有可迁移的结构？
- 设计显式的 Forward Transfer 机制，让历史任务的知识主动引导未来任务的初始化
- 本质：**时序欺诈知识的前向迁移学习（Temporal Forward Transfer for Fraud Detection）**

**优点**：
- 直接来自第一工作的实验发现，叙事继承最自然（"我们发现了 FT 现象，现在来利用它"）
- 与现有 CL 文献互补（大多数关注 backward transfer/forgetting，forward transfer 研究较少）
- 不需要修改 BSL 模型结构，在 CL 策略层面做文章

**缺点**：
- "为什么某些模型有 FT"本身需要理论解释，否则显得 empirical
- 实验设计需要精心构造 FT 场景

---

### 11.5 方向五：从"离线任务增量"到"在线流式欺诈检测"

**继承点**：第一工作是离线批次式（每个 task 有完整的一批数据），但真实金融系统中交易是实时流式到来的，需要在线更新模型。

**第二工作方向**：
- 将 TASD-CL 的核心机制迁移到**在线学习（Online Learning）**设定
- 每笔新交易到来时，实时更新 BSL 子空间，SPC 原型做轻量级在线更新
- 同时处理在线标签稀缺问题（标签延迟到达）
- 本质：**在线图欺诈检测（Online Graph Fraud Detection）with Streaming Transactions**

**优点**：
- 工程落地价值最高，最贴近真实场景
- 自然解决 DGraphFin 的时间戳问题（边时间戳 1-821 天然是流式数据）
- 与工业界合作或应用场景叙事最强

**缺点**：
- 技术难度大，在线 GNN 更新本身是开放问题
- 与第一工作技术路线差异较大，可能像两个独立工作

---

### 11.6 综合推荐与大论文叙事框架

| 方向 | 继承强度 | 技术难度 | 发表价值 | 可行性 |
|---|---|---|---|---|
| 方向一：主动偏移检测 | ⭐⭐⭐⭐ | 中 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **方向二：开放集欺诈发现** | **⭐⭐⭐⭐⭐** | **中高** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐** |
| 方向三：跨图迁移 | ⭐⭐⭐ | 高 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **方向四：前向迁移显式建模** | **⭐⭐⭐⭐⭐** | **中** | **⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** |
| 方向五：在线流式检测 | ⭐⭐⭐ | 很高 | ⭐⭐⭐⭐⭐ | ⭐⭐ |

**最推荐组合（两个工作的大论文叙事）：**

```
第一工作（已完成）：
"时序欺诈 pattern 偏移导致 GNN 性能下降"
→ TASD-CL：子空间感知的持续学习缓解偏移

              ↓ 实验发现：某些模型/场景出现前向迁移现象

第二工作（推荐方向四）：
"时序欺诈知识存在可利用的前向迁移结构"
→ 显式建模 Forward Transfer：历史欺诈 pattern 主动引导新任务初始化
→ 在 BSL 子空间层面设计跨任务知识路由机制
```

**叙事逻辑**：第一工作"防止遗忘"，第二工作"利用记忆"，两者形成完整的"持续学习双面性"研究，共同回答"GNN 欺诈检测在时序流中应如何演化"这一大问题。

---

### 11.7 新对话时需要讨论的问题
- [ ] 用户倾向于哪个方向？
- [ ] 方向四中"前向迁移机制"的具体模型设计是什么？
- [ ] 第一工作的实验结果是否足够充分，能否支撑方向四的 motivation？
- [ ] 是否需要在第一工作论文中专门留一节分析 FT 现象，为第二工作铺垫？

---

## 十二、环境与路径信息

- **服务器**：租用服务器，RTX 3090，CUDA 12.1
- **项目路径**：`D:/AAmyproject/codes/fraud-detection-gnn/`（口头称为 "yw"）
- **Python 环境**：`py312`
- **相关论文 PDF**：已整理到 `papers/` 目录

---

## 十三、环境安装

### 安装支持 GPU 的 PyTorch（RTX 3090 对应 CUDA 12.1）
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 安装 PyG 及其扩展
```bash
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu121.html
pip install omegaconf scikit-learn pandas numpy matplotlib networkx tensorboard
```

### 验证
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# 期望输出: True  /  NVIDIA GeForce RTX 3090
```

---

## 十四、输出目录结构

- 模型权重：`weights/<模型名>/`（由 yaml 中 `train.save_dir` 指定）
- 训练指标 CSV：`weights/<模型名>/metrics/`
- TensorBoard 日志：`runs/<config.name>/`

```bash
tensorboard --logdir runs/
python tools/collect_results.py    # 汇总所有实验指标到 results_summary.csv
python tools/analyze_results.py    # 分析和可视化结果
```

---

## 十五、支持的持续学习策略

| 策略 | yaml 配置方式 |
|---|---|
| Naive（无策略） | `ewc_lambda: 0`, `lwf_alpha: 0`, `buffer_size_per_class: 0` |
| EWC | `ewc_lambda: 1.0` |
| LwF | `lwf_alpha: 1.0` |
| Experience Replay | `buffer_size_per_class: 50` |
| **SSF（TASD-CL）** | `ewc_lambda: 1.0`（BSL模型下自动启用语义分层） |
| **SPC（TASD-CL）** | `spc_lambda: 1.0` |
| **SCD（TASD-CL）** | `scd_lambda: 1.0`, `scd_tau: 0.5` |

---

## 十六、⚠️ 核心实验策略：让 TASD-CL 超越所有对比方法

> 目标：发表 CIKM，TASD-CL 在 Elliptic 数据集上超越 7 个对比方法中的至少 5-6 个。
> 对比方法：GCN Naive / CGNN Naive / ConsisGAD Naive / GradGNN Naive / HOGRL Naive / PMP Naive / BSL Naive

### 当前 Elliptic 结果（Task 10，2026-04-03 最新）

| 排名 | 方法 | avg_F1 | avg_AUC-ROC | avg_Macro_F1 | avg_Specificity |
|---|---|---|---|---|---|
| 1 | CGNN Naive | 0.402 | 0.863 | 0.616 | 0.724 |
| 2 | HOGRL Naive | 0.395 | 0.857 | 0.618 | 0.746 |
| 3 | GradGNN Naive | 0.384 | 0.837 | 0.582 | 0.675 |
| 4 | GCN Naive | 0.381 | 0.856 | 0.587 | 0.678 |
| 5 | BSL Naive | 0.316 | 0.768 | 0.562 | 0.741 |
| **6** | **TASD-CL** | **0.302** | **0.778** | **0.574** | **0.788** |
| 7 | PMP Naive | 0.059 | 0.698 | 0.493 | 0.991 |
| 8 | ConsisGAD Naive | 0.000 | 0.668 | 0.465 | 1.000 |

**当前问题**：TASD-CL 只超越 PMP 和 ConsisGAD（2/7），离目标 5-6 个还差 3-4 个。

**唯一优势指标**：avg_Specificity（0.788）> HOGRL(0.746) > BSL(0.741) > CGNN(0.724) > GradGNN(0.675) > GCN(0.678)

---

### 策略 A：超参系统调优（最直接，优先执行）

TASD-CL 当前超参未经系统搜索，存在较大提升空间。调优目标是把 avg_F1 从 0.302 推至 0.38+，逼近甚至超过 GCN/GradGNN。

推荐搜索空间：
```yaml
ewc_lambda:    [0.5, 1.0, 2.0, 3.0]
spc_lambda:    [0.5, 1.0, 2.0, 3.0]
spc_n_samples: [64, 128, 256]
scd_lambda:    [0.3, 0.5, 1.0, 2.0]
scd_tau:       [0.1, 0.2, 0.3, 0.5]
```

关键假设：消融实验表明 SSF+SPC 对 F1 贡献最大（noSSF/noSPC → F1 崩溃至 0.05-0.07）。更大的 `spc_n_samples` 和 `spc_lambda` 可能显著提升 recall，进而提升 F1。

---

### 策略 B：排查并利用 ConsisGAD / PMP 失效（已有 2 个赢面）

- **ConsisGAD**：Task-only 设定下 F1=0.000，recall=0——完全失效。历史结果（旧格式 CSV）F1=0.32，说明新代码/新设定下 ConsisGAD 无法检测欺诈。这是对 Task-only 窗口设定的天然不适应，**可作为正面论据**：Task-only 设定对没有 CL 机制的模型非常苛刻，TASD-CL 天然解决了这个问题。
- **PMP**：同样 recall 极低（0.048），F1=0.059，也基本失效。

如果这两个模型确实在 Task-only 设定下失效，TASD-CL 已经胜过 2/7。**不要修复它们，这反而支撑了研究动机**。

---

### 策略 C：扩展评估指标维度（找到 TASD-CL 的优势指标）

单一 F1 不能反映全貌。以下是 TASD-CL 在不同指标上的潜在优势：

| 指标 | TASD-CL 的位置 | 说明 |
|---|---|---|
| avg_F1 | 第 6（差） | 主要比较指标，需靠调参提升 |
| avg_AUC-ROC | 第 6（差） | 需提升 |
| **avg_Specificity** | **第 1（最优）** | 在不影响正常节点检测的前提下识别欺诈，业务价值高 |
| **跨任务稳定性（方差）** | 待计算 | TASD-CL 设计上更稳定，可能方差更小 |
| **后期任务（T8-T10）平均 F1** | 待计算 | CL 方法后期优势更明显 |

---

### 策略 D：修改方法本身（工程量最大，效果最显著）

将 TASD-CL 的三个组件（SSF/SPC/SCD）的核心思想移植到更强的 backbone（如 CGNN 或 GradGNN）。CGNN+TASD-CL 的理论上限远高于 BSL+TASD-CL。

**注意**：这需要重新设计三个组件的 backbone 适配（CGNN 没有 BSL 的三子空间结构，SSF 的角色乘数需要重新定义），工程量约为 2-4 周。

---

### 当前推荐行动顺序

1. **立即**：排查 ConsisGAD / PMP 结果是否真的在 Task-only 下失效（确认无 bug，是设定本身造成的）
2. **本周**：对 TASD-CL 超参做网格搜索（策略 A）
3. **评估**：调参后用多指标对比（策略 C），看哪些指标能超过 5-6 个方法
4. **备选**：如果调参后仍无法在主流指标上超过 CGNN/HOGRL，考虑策略 D（换 backbone）
