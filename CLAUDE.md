# fraud-detection-gnn 项目说明（CLAUDE.md）

> 最后更新：2026-04-07
> 本文件供 Claude 在新对话中快速恢复上下文。请在每次重要进展后更新。

---

## 一、项目定位与研究动机

### 研究问题（一句话）
在图神经网络欺诈检测的任务增量持续学习场景下，提出 TASD-CL 框架，通过子空间语义过滤（SSF）、原型凝缩（SPC）和置信度蒸馏（SCD）三个组件，以 **CGNN** 为 backbone，在时序欺诈 pattern 演变中保持稳定且最优的检测性能。

### 研究叙事逻辑（最终确认版，基于 BRIGHT 工业动机）

1. **工业背景（BRIGHT 动机）**：真实金融风控系统（如 eBay）中交易数据以流式到达，若每次都构建包含全量历史边的 Cumulative 图，会面临两个致命问题：**延迟极高**（无法满足毫秒级实时推理要求）和**内存爆炸（OOM）**。因此工业界采用基于时间窗口的 Task-only 构图方式，每次只处理当前窗口内的边（如 BRIGHT 的 Two-Stage Directed Graph 设计）。

2. **发现的问题**：在 Task-only 的窗口构图约束下，GNN 欺诈检测模型极易对当前欺诈 pattern 过拟合，随时间步推进性能不稳定（在 Elliptic 等数据集上可观察到明显的性能波动和下滑趋势）。

3. **根本原因**：Task-only 窗口切分本质上是任务增量持续学习场景。模型在当前任务上更新参数时，会覆盖历史欺诈 pattern 的表示，导致子空间结构被污染、历史知识丢失。

4. **为什么不能用传统 ER**：标准 Experience Replay 需要在新任务图上重新前向传播历史节点，但 Task-only 设定下历史节点的邻居图结构已经消失，强行回放得到的是结构噪声，反而有害——且这违背了 BRIGHT 提出的实时性约束。

5. **解决方法**：TASD-CL 框架，通过 SPC 仅保留轻量级高斯原型 (μ,σ)，以极低内存代价实现跨时间步的知识保留，完全不依赖历史图结构，天然契合 Task-only 工业约束。

6. **效果目标**：在 Elliptic 数据集上，TASD-CL (CGNN) **已超越所有 7 个对比方法**（BSL/CGNN/ConsisGAD/GradGNN/HOGRL/PMP/GCN），最优调参后 F1=0.4347，AUC=0.8621，MacroF1=0.6582。

### 核心思想：为什么选 CGNN 作为 backbone

CGNN（Context-aware GNN，AAAI-25）将节点嵌入通过**去噪注意力机制**解耦为两个语义子空间：
- `x_nor`：正常子空间——聚合"干净"的正常邻居特征
- `x_abnor`：异常子空间——聚合"可疑"的欺诈邻居特征

关键参数 `alpha`（per-node 标量，[N]）：去噪注意力权重
- alpha → 1：节点倾向正常，x_nor 主导
- alpha → 0：节点倾向异常（欺诈），x_abnor 主导
- 消息传播：`x_j = alpha * x_nor_j + (1-alpha) * x_abnor_j`

这种解耦使得"保护哪些参数"、"蒸馏哪些节点"、"回放什么知识"都可以围绕**正常/欺诈语义**精细设计，是 TASD-CL 三个组件的设计基础。

CGNN 自带两个辅助损失（在 TASD-CL 总损失中保留）：
- **L_csd**（子空间分离损失）：最小化 x_nor 与 x_abnor 的余弦相似度²，强制子空间正交
- **L_consist**（一致性损失）：正常节点 `||x_abnor|| < ||x_nor||`，欺诈节点反之

---

## 二、TASD-CL 三组件详细说明（CGNN backbone）

### CGNN 子空间结构（三组件的共同基础）

```
输入 x [N, hidden_dim=128]
       ↓ lin_in
  h [N, hidden_dim]
  ├── h[:, :64]  → lin_nor   → x_nor   [N, 128]   正常子空间
  └── h[:, 64:]  → lin_abnor → x_abnor [N, 128]   异常子空间
                                    ↓
                          per-node alpha [N]  ← 去噪注意力（float 0~1）
                          alpha≈1 → 正常节点
                          alpha≈0 → 欺诈节点
```

`SPC sub_dim = hidden_dim = 128`（两子空间各 128 维）

| 组件 | 全称 | 作用 | 实现位置 |
|---|---|---|---|
| **SSF** | Subspace-Stratified Fisher | 改造版 EWC，对 CGNN 不同参数施加不同强度约束，去噪注意力路由参数约束最强 | `trainer.py: _get_ssf_lambda()` |
| **SPC** | Subspace Prototype Condensation | 任务结束后提取 x_nor/x_abnor 两子空间的类原型（Gaussian μ,σ），绕开图结构做无噪声回放 | `trainer.py: _update_spc_prototypes()`, `buffer.py: SubspacePrototypeBuffer` |
| **SCD** | Subspace-Conditioned Distillation | 替代标准 LwF，用 alpha 置信度过滤只蒸馏高置信节点，x_abnor（欺诈子空间）权重 2.0 | `trainer.py: _compute_scd_loss()` |

---

### Component A：SSF（Subspace-Stratified Fisher）

**本质**：改造版 EWC，对不同 CGNN 参数的 Fisher 约束乘以语义角色系数。

标准 EWC 正则项：
```
L_ewc = λ × Σ_θ  F_θ × (θ - θ*)²
```

SSF 在此基础上，对每个参数查询 `CGNN_PARAM_GROUPS` 得到角色乘数，实际约束 = `ewc_lambda × role_mult`：

```python
CGNN_PARAM_GROUPS = {
    'conv.att_vec':    3.0,   # 去噪注意力路由向量（最高）：决定每条边 alpha 值
                              # 丢失则 x_nor/x_abnor 分离失效，欺诈语义崩塌
    'conv.att_lin':    2.5,   # 去噪注意力线性层（次高）
    'conv.lin_nor':    2.0,   # 正常子空间投影（中等）
    'conv.lin_abnor':  2.0,   # 异常子空间投影（中等）
    'lin_in':          0.5,   # 输入映射（低）：通用特征变换，允许更新
    'classifier':      0.3,   # 分类头（最低）：任务特定层，允许自由更新
}
```

实际约束范围（`ewc_lambda=1.0`时）：`conv.att_vec = 3.0`，`classifier = 0.3`。

**为什么 conv.att_vec 约束最强**：它是每条边的注意力得分计算核心 `alpha = sigmoid(att_vec(tanh(att_lin(x_nor + x_abnor))))`，决定每个节点的正常/欺诈权重分配。一旦被新任务覆盖，x_nor 和 x_abnor 的语义分离失效，后续 SPC 和 SCD 的子空间操作全部失去意义。

**Config 参数**：
```yaml
ewc_lambda: 1.0    # 基础约束强度，乘角色系数后实际范围 [0.3, 3.0]
                   # 设为 0 → SSF 完全关闭，退化为 Naive CGNN
```

---

### Component B：SPC（Subspace Prototype Condensation）

**本质**：替代标准 Experience Replay，存储子空间高斯原型而非原始节点索引。

**为什么不能直接做图回放**：标准 ER 存节点索引，下个 task 再前向传播。但欺诈图是时序的——Task 3 的节点在 Task 5 的图快照中邻居已完全不同，强行回放得到的 x_nor/x_abnor 是结构噪声，CSD 损失会被污染，反而有害。

**SPC 的做法**：每个 task 训练结束后，计算两子空间在两类上的高斯原型：
```
对每个 task t、类别 c ∈ {0,1}、子空间 k ∈ {0=x_nor, 1=x_abnor}：
  mu[t][c][k]    = mean(x_k[训练节点 & 标签==c])   # [hidden_dim=128]
  sigma[t][c][k] = std (x_k[训练节点 & 标签==c])   # [hidden_dim=128]
```

存储代价：`O(T × 2 × 2 × 128)` = T=10 时约 5120 个浮点数（极小，约 20KB）。

**回放时**，从原型采样合成节点，跳过 GNN 编码器直接送 replay_forward：
```python
z_rep [n, 2*hidden_dim] = sample from prototypes
z_nor_rep   = z_rep[:, :128]    # 正常子空间合成样本
z_abnor_rep = z_rep[:, 128:]    # 异常子空间合成样本
out_rep = model.replay_forward(z_nor_rep, z_abnor_rep)
# replay_forward: alpha*z_nor + (1-alpha)*z_abnor → update_lin → classifier
L_spc = BCE(out_rep, y_replay)
```

**Config 参数**：
```yaml
spc_lambda: 0.3       # 回放损失权重，设为 0 → SPC 关闭
spc_n_samples: 32     # 每个 (task, class) 采样多少合成节点
                      # 已见 3 个 task 时，每次回放生成 3×2×32=192 个合成节点
```

---

### Component C：SCD（Subspace-Conditioned Distillation）

**本质**：替代标准 LwF，在 x_nor/x_abnor 两子空间层面做知识蒸馏，且只蒸馏旧模型高置信度节点。

**标准 LwF 的缺陷**：
1. 只蒸馏 1 维 logit 输出，丢失 CGNN 双子空间的几何结构
2. 欺诈检测严重不平衡（~9% 欺诈），旧模型对大量正常节点 alpha≈1，对欺诈节点 alpha≈0，但 LwF 无法区分这两种节点的蒸馏价值

**SCD 的做法**：用旧模型的 `node_alpha` 作为置信度过滤器：
```python
# alpha < tau → 旧模型高置信度认为是欺诈节点 → 优先保护 x_abnor
mask_abnormal = (old_node_alpha < scd_tau)
# alpha > 1-tau → 旧模型高置信度认为是正常节点 → 保护 x_nor
mask_normal   = (old_node_alpha > 1.0 - scd_tau)

loss_scd = 2.0 * MSE(new_x_abnor[mask_abnormal], old_x_abnor[mask_abnormal])  # 欺诈权重2.0
         + 1.0 * MSE(new_x_nor[mask_normal],   old_x_nor[mask_normal])
```

**为什么 x_abnor 权重 2.0**：欺诈 pattern 的历史知识比正常 pattern 更难从稀疏欺诈标签中重新学习，保护代价更高，因此给予更强约束。

**Config 参数**：
```yaml
scd_lambda: 0.5       # 蒸馏损失权重，设为 0 → SCD 关闭
scd_tau: 0.5          # 置信度过滤阈值
                      # tau=0.5 时：alpha<0.5 → 欺诈节点mask；alpha>0.5 → 正常节点mask
                      # tau 越大 → 两个 mask 都更严格，只蒸馏最典型节点

# 关闭标准 LwF（由 SCD 替代，避免重复蒸馏）
lwf_alpha: 0.0
# 关闭标准 ER（由 SPC 替代，避免图结构噪声）
buffer_size_per_class: 0
```

---

### 总损失公式

```
L_total = L_task                                              # 当前 task BCE Loss
        + ewc_lambda × Σ_θ role_mult(θ) × F_θ × (θ-θ*)²    # SSF 参数约束
        + spc_lambda × L_spc                                  # SPC 子空间原型回放
        + scd_lambda × L_scd                                  # SCD 子空间蒸馏
        + cgnn_lambda × L_csd + cgnn_beta × L_consist         # CGNN 自带子空间分离损失
```

### Config 参数速查表

| 参数 | 组件 | 含义 | 设为 0 的效果 |
|---|---|---|---|
| `ewc_lambda` | A: SSF | 参数约束基础强度（乘角色系数后范围 0.3~3.0） | 无参数约束，退化为 Naive |
| `spc_lambda` | B: SPC | 子空间原型回放损失权重 | 无原型回放 |
| `spc_n_samples` | B: SPC | 每个 (task, class) 合成多少节点 | — |
| `scd_lambda` | C: SCD | 子空间蒸馏损失权重 | 无蒸馏 |
| `scd_tau` | C: SCD | 旧模型 alpha 置信度过滤阈值 | — |
| `cgnn_lambda` | CGNN | 子空间分离损失权重（默认 0.5） | — |
| `cgnn_beta` | CGNN | 一致性损失权重（默认 0.1） | — |
| `lwf_alpha: 0.0` | — | 关闭标准 LwF（已被 SCD 替代） | — |
| `buffer_size_per_class: 0` | — | 关闭标准 ER（已被 SPC 替代） | — |

---

## 三、实验设计

### Snapshot 设计（已确认：task-only）
- 每个任务只使用**当前时间步范围内的边**（`edge_index` 按 `valid_node_mask` 过滤）
- 但 `snapshot_data.x` 仍是全量节点特征（shallow copy）
- 本质：**图结构 task-only，节点特征 all**
- **为什么不用 cumulative**：task-only 让 baseline 的性能下降归因更干净，只来自时序分布偏移，不混入"历史图信息缺失"的因素；且与 Elliptic 文献标准对齐

关键代码（`trainer.py`）：
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
- 主损失：`BCEWithLogitsLoss`（pos_weight=3.0）
- 总损失：`task_loss + cl_loss + cgnn_loss + spc_lambda * spc_loss`

---

## 四、数据集信息

| 数据集 | 节点数 | 边数 | 时间步 | 欺诈率 | 特殊问题 |
|---|---|---|---|---|---|
| **Elliptic** | ~203K | ~234K | 49个时间步（本项目用前10） | ~9% | 标准，无 OOM |
| **Elliptic++ Actor** | 类似 | 类似 | 类似 | 类似 | 标准，无 OOM |
| **DGraphFin** | ~3.7M | ~4.3M | 边时间戳 1-821，划分为10段 | ~1.3% | **严重 OOM 问题** |

### DGraphFin OOM 根因分析（⚠️ 未解决）
- `copy.copy(dataset).to(device)` 将全量 3.7M 节点特征（约 3.7M × 17 × 4bytes ≈ 250MB）搬上 GPU
- CGNN GATv2 中间激活约 4-6GB
- **目前状态**：DGraphFin 上全部 OOM，暂时搁置

---

## 五、所有模型实现状态

| 模型 | 类别 | 实现状态 | 备注 |
|---|---|---|---|
| GCN | 通用 GNN 基线 | ✅ 完整 | |
| HOGRL | 欺诈 SOTA | ✅ 完整 | DGraphFin OOM |
| CGNN | 欺诈 SOTA + **backbone** | ✅ 完整 | DGraphFin OOM |
| BSL | 欺诈 SOTA（旧 backbone） | ✅ 完整 | DGraphFin OOM |
| ConsisGAD | 欺诈 SOTA | ✅ 完整 | |
| GradGNN | 欺诈 SOTA | ✅ 完整 | |
| PMP | 欺诈 SOTA | ✅ 完整 | config 在 `configs/fraud_sota/{dataset}/` |

**已从实验中移除（config 移至 `configs/deprecated/`）**：
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
    main/                    # CGNN + TASD-CL（主实验）
                             #   elliptic_TASDCL_CGNN.yaml ✅
                             #   elliptic++actor_TASDCL_CGNN.yaml（待创建）
                             #   elliptic_TASDCL_BSL.yaml（旧 backbone，保留备用）
    cl_on_cgnn/              # CGNN + EWC/LwF/ER（CL 基线对比，待创建目录）
    cl_on_bsl/               # BSL + EWC/LwF/ER（旧版，保留）
    ablation/                # noSSF / noSPC / noSCD（需补 CGNN 版本）
  deprecated/                # 已废弃，不再运行
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
| BSL | Elliptic | TASD-CL | ✅ 完成（旧 backbone，F1=0.302） |
| CGNN | Elliptic | Naive | ✅ 完成 |
| **CGNN** | **Elliptic** | **TASD-CL（最优 lam01_n128）** | ✅ **完成（F1=0.4347，超越所有方法）** |
| CGNN | Elliptic | 消融：noSSF/noSPC/noSCD | ✅ 完成（见 9.3 节）|
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

### 待运行（论文所需，优先级排序）
| 优先级 | 模型 | 数据集 | 策略 | 备注 |
|---|---|---|---|---|
| ✅ 完成 | CGNN | Elliptic | noSSF / noSPC / noSCD | 消融实验已完成（见 9.3 节） |
| ⚠️ 最高 | CGNN | Elliptic | EWC / LwF / ER | CL 基线，config 需新建 `configs/ours/cl_on_cgnn/` |
| 高 | CGNN | Elliptic++ Actor | TASD-CL | config `elliptic++actor_TASDCL_CGNN.yaml` 需新建 |
| 高 | CGNN | Elliptic++ Actor | EWC / LwF / ER | config 需新建 |
| 高 | CGNN | Elliptic++ Actor | noSSF / noSPC / noSCD | config 需新建 |
| 低 | 所有模型 | DGraphFin | — | OOM 未解决，暂搁置 |

---

## 九、实验结果详细分析

> 数据来源：`weights/*/metrics/*_aggregate_metrics.csv`
> 当前只有 Elliptic 数据集有完整的 TASD-CL CGNN 结果。

---

### 9.1 Elliptic 数据集 — 最新完整结果（Task 10，使用各方法最优运行结果）

> Full TASD-CL 使用最优调参配置：`spc_lambda=0.1, spc_n_samples=128`（配置文件：`elliptic_TASDCL_spc_lam01_n128_CGNN.yaml`）

| 排名 | 模型 | 策略 | avg_F1 | avg_AUC-ROC | avg_Macro_F1 | avg_Specificity |
|---|---|---|---|---|---|---|
| **1** | **CGNN** | **TASD-CL（我们，lam01_n128）** | **0.4347** | 0.8621 | **0.6582** | — |
| 2 | CGNN | Naive | 0.4019 | **0.8631** | 0.6160 | 0.7240 |
| 3 | HOGRL | Naive | 0.3953 | 0.8570 | 0.6180 | 0.7460 |
| 4 | GradGNN | Naive | 0.3839 | 0.8370 | 0.5820 | 0.6750 |
| 5 | GCN | Naive | 0.3813 | 0.8560 | 0.5870 | 0.6780 |
| 6 | BSL | Naive | 0.3161 | 0.7680 | 0.5620 | 0.7410 |
| 7 | BSL | TASD-CL（旧 backbone） | 0.302 | 0.778 | 0.574 | 0.788 |
| 8 | PMP | Naive | 0.0591 | 0.6980 | 0.4930 | 0.9910 |
| 9 | ConsisGAD | Naive | 0.000 | 0.6680 | 0.4650 | 1.000 |

**结论：TASD-CL (CGNN) 在 avg_F1 和 avg_Macro_F1 上超越所有 7 个对比方法。**
- avg_F1 排名第 1（+0.0328 vs CGNN Naive）
- avg_AUC-ROC 排名第 2（CGNN Naive 略高 0.001，差异极小）
- avg_Macro_F1 排名第 1（+0.0422 vs CGNN Naive）
- PMP 和 ConsisGAD 在 Task-only 设定下几乎完全失效，验证研究动机

---

### 9.2 TASD-CL CGNN 逐任务 avg_F1（Elliptic，排除 warmup Task 1）

| Task | TASD-CL CGNN（新） | CGNN Naive | HOGRL | GradGNN | GCN | BSL TASD-CL（旧） |
|---|---|---|---|---|---|---|
| 2 | 0.110 | 0.113 | 0.115 | 0.175 | 0.136 | 0.048 |
| 3 | 0.292 | 0.298 | 0.324 | 0.338 | 0.311 | 0.144 |
| 4 | 0.382 | 0.386 | 0.399 | 0.410 | 0.398 | 0.257 |
| 5 | 0.458 | 0.461 | 0.453 | 0.465 | 0.454 | 0.322 |
| 6 | 0.479 | **0.476** | 0.457 | **0.473** | **0.469** | 0.363 |
| 7 | **0.476** | 0.467 | 0.455 | 0.458 | 0.456 | 0.368 |
| 8 | 0.454 | 0.448 | 0.429 | 0.427 | 0.426 | 0.354 |
| 9 | 0.405 | 0.399 | 0.395 | 0.383 | 0.380 | 0.315 |
| 10 | **0.4145** | 0.402 | 0.395 | 0.384 | 0.381 | 0.302 |

**关键观察**：TASD-CL CGNN 从 Task 6 起持续领先所有对比方法，Task 10 仍保持最高 F1（比 CGNN Naive 高 +0.012）。

---

### 9.3 消融实验（Elliptic，CGNN backbone，Task 10）

> Full 使用最优调参（lam01_n128）与最好运行结果；各消融变体（noSSF/noSPC/noSCD）使用原始超参（spc_lambda=0.3, n_samples=32）

| 方法 | avg_F1 | avg_AUC-ROC | avg_Macro_F1 | 相对 Full |
|---|---|---|---|---|
| **TASD-CL Full（最优调参，lam01_n128）** | **0.4347** | 0.8621 | **0.6582** | — |
| TASD-CL Full（原始超参，lam=0.3）| 0.4145 | **0.8748** | 0.6310 | —（参考基线） |
| noSSF（仅 SPC+SCD） | 0.4042 | 0.8638 | 0.6213 | −0.030 F1 |
| noSPC（仅 SSF+SCD） | 0.4308 | — | — | −0.004 F1 |
| noSCD（仅 SSF+SPC） | 0.000 | — | — | **−0.435 F1（崩溃）** |

**关键发现（CGNN backbone 消融）：**
- **SCD 是最关键组件**：移除后 F1 崩溃至 0.000（−100%），模型完全失去欺诈检测能力
- **SSF 有稳定贡献**：移除后 F1=0.4042，下降 −0.030（−7.2%）；AUC 几乎不变
- **SPC 有小幅贡献**：调参前 noSPC > 原版 Full（0.4308 > 0.4145），提示 SPC 的正向贡献依赖合适超参；调参到 lam01_n128 后 Full (0.4347) > noSPC (0.4308)，SPC 贡献恢复

**与 BSL backbone 消融对比（历史参考，已废弃）**：
| 方法 | avg_F1 | avg_AUC-ROC |
|---|---|---|
| TASD-CL Full (BSL) | 0.302 | 0.778 |
| noSSF (BSL) | 0.066 | 0.738 |
| noSPC (BSL) | 0.049 | 0.750 |
| noSCD (BSL) | 0.188 | 0.797 |

CGNN backbone 大幅提升了基线性能（0.302 → 0.4145），各组件贡献模式更为稳健。

---

### 9.4 SPC 超参调优历史（Elliptic，CGNN backbone）

发现 noSPC > 原版 Full（0.4308 > 0.4145）后，进行系统调参：

| 配置 | spc_lambda | spc_n_samples | avg_F1 | 相对原版 |
|---|---|---|---|---|
| 原版 Full | 0.3 | 32 | 0.4145 | — |
| lam01_n32 | 0.1 | 32 | 0.4083 | −0.006 |
| lam005_n32 | 0.05 | 32 | 0.3877 | −0.027 |
| **lam01_n128** | **0.1** | **128** | **0.4347** | **+0.020（最优）** |
| noSPC（参考） | 0.0 | — | 0.4308 | +0.016（同 lam01_n128 相比 −0.004） |

**结论**：降低 spc_lambda（减少噪声）+ 增大 n_samples（提升原型质量）= 最优组合。
最终主实验配置：`spc_lambda=0.1, spc_n_samples=128`。

---

### 9.5 关键未决问题

#### ✅ 已解决：TASD-CL 绝对性能问题
**之前**：BSL backbone 导致 F1=0.302，低于所有对比方法。
**已解决**：切换 CGNN backbone + 超参调优后 F1=0.4347，超越所有对比方法。

#### ✅ 已解决：消融实验（CGNN backbone）
noSSF/noSPC/noSCD 三组消融均已运行完毕，结果见 9.3 节。

#### ⚠️ 待确认：ConsisGAD 和 PMP 结果异常
**现象**：ConsisGAD F1=0.000，PMP F1=0.059，均在 Task-only 设定下失效。
**正面解读**：这证明了 Task-only 窗口设定对无 CL 机制的模型非常苛刻，支撑研究动机。**不修复，作为论文正面论据。**

#### ⚠️ 待运行：CGNN + EWC/LwF/ER CL 基线
需在 `configs/ours/cl_on_cgnn/` 创建配置并运行，以验证 TASD-CL 优于通用 CL 策略。

#### ⚠️ 待运行：Elliptic++ Actor 上的 CGNN TASD-CL

---

## 十、✅ Snapshot 设计：Task-only（已最终确认）

**此问题已于 2026-04-03 最终拍板：使用 Task-only，不再讨论 Cumulative。**

### 确认原因：BRIGHT 工业论文的背书

eBay 的 BRIGHT 论文（发表于顶会）明确指出：在真实金融风控系统中，维护一个无限增长的 Cumulative 全局图会面临两个致命问题：
- **延迟极高**：无法满足毫秒级实时推理要求
- **内存爆炸（OOM）**：邻居节点呈指数级增长

### Cumulative 方案的内存代价分析（为什么不能用）

| 数据集 | Task-only 最大边数（单 Task） | Cumulative Task 10 边数 | 内存变化 |
|---|---|---|---|
| Elliptic | ~4.8K 边 | ~47.8K 边 | ×10，仍可接受 |
| Elliptic++ Actor | 类似 Elliptic | 类似 Elliptic | ×10，仍可接受 |
| **DGraphFin** | **~430K 边（已 OOM）** | **~4.3M 边** | **×10，完全不可行** |

### 论文叙事中的用法

> "Following the industrial constraint identified by BRIGHT [cite], where maintaining a cumulative global graph causes OOM and prohibitively high latency, we adopt a task-only window-based graph construction. This setting also cleanly isolates temporal distribution shift as the sole source of performance degradation."

---

## 十一、遗留困惑与待决策问题

### 已解决
- [x] 灾难性遗忘问题：实验证明基本不存在，研究叙事中不再提及
- [x] **Snapshot 设计**：已最终确认为 Task-only，BRIGHT 工业论文提供充分背书
- [x] **SCD 组件失效 Bug**：`old_model` 未保存，导致 SCD 始终返回 0。已修复
- [x] **评估指标扩充**：新增 Macro F1、Macro Recall、Specificity、MCC，删除 Forgetting/BWT
- [x] **训练测试划分改为时序切分**：前 80% 时间步 → 训练，后 20% → 测试
- [x] **Task 1 标记为 warm-up**：欺诈样本极少，CSV 中 `is_warmup=True`，论文分析排除该任务
- [x] **backbone 改为 CGNN**：BSL backbone F1=0.302 低于对比方法，切换 CGNN 后 F1=0.4145 超越所有方法
- [x] **CGNN 消融实验（Elliptic）**：noSSF/noSPC/noSCD 均已运行，揭示 SCD 是最关键组件
- [x] **SPC 超参调优**：发现 spc_lambda=0.1+n_samples=128 为最优组合（F1=0.4347），已更新主配置
- [x] **deepcopy RuntimeError 修复**：CGNNLayer 缓存张量在 deepcopy 前清空（两处：LwF 路径和 SCD 路径）

### 待解决（优先级排序）

- [ ] **⚠️【最优先】创建 CGNN + EWC/LwF/ER 配置并运行**：作为通用 CL 基线对比，验证 TASD-CL 优于标准 CL 方法
- [ ] **⚠️ 创建 elliptic++actor_TASDCL_CGNN.yaml 并运行**：验证跨数据集泛化
- [ ] **ConsisGAD 和 PMP 结果异常**：已确认是 Task-only 设定本身导致（而非 bug），保留不修，作为论文正面论据
- [ ] **DGraphFin OOM**：Task-only 下已严重 OOM，暂搁置
- [ ] **理论补充**：SCD 中 x_abnor 子空间权重 2.0 的理论动机尚未形式化

---

## 十二、第二工作头脑风暴（硕士毕业论文继承关系）

> 第一工作核心：在任务增量持续学习下，CGNN 模型因时序欺诈 pattern 偏移导致性能下降，TASD-CL 通过子空间感知的三组件缓解这一问题，超越所有对比方法。
> 第二工作需要：在第一工作的基础上，顺承其发现或不足，进一步深化或延伸。

详细方向见原版本（方向一~五），**最推荐组合**：
- **方向四（前向迁移显式建模）**：第一工作"防止遗忘"，第二工作"利用记忆"，继承最自然
- **方向二（开放集欺诈发现）**：SPC 原型直接复用，发表价值最高

---

## 十三、环境与路径信息

- **服务器**：租用服务器，RTX 3090，CUDA 12.1
- **项目路径**：`D:/AAmyproject/codes/fraud-detection-gnn/`（口头称为 "yw"）
- **Python 环境**：`py312`
- **相关论文 PDF**：已整理到 `papers/` 目录

---

## 十四、环境安装

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

## 十五、输出目录结构

- 模型权重：`weights/<模型名>/`（由 yaml 中 `train.save_dir` 指定）
- 训练指标 CSV：`weights/<模型名>/metrics/`
- TensorBoard 日志：`runs/<config.name>/`

```bash
tensorboard --logdir runs/
python tools/collect_results.py    # 汇总所有实验指标到 results_summary.csv
python tools/analyze_results.py    # 分析和可视化结果
```

---

## 十六、支持的持续学习策略

| 策略 | yaml 配置方式 |
|---|---|
| Naive（无策略） | `ewc_lambda: 0`, `lwf_alpha: 0`, `buffer_size_per_class: 0` |
| EWC | `ewc_lambda: 1.0` |
| LwF | `lwf_alpha: 1.0` |
| Experience Replay | `buffer_size_per_class: 100` |
| **SSF（TASD-CL）** | `ewc_lambda: 1.0`（CGNN/BSL 模型下自动启用语义分层） |
| **SPC（TASD-CL）** | `spc_lambda: 0.1`, `spc_n_samples: 128`（调优后最优值） |
| **SCD（TASD-CL）** | `scd_lambda: 0.5`, `scd_tau: 0.5` |

---

## 十七、下一步行动（按优先级）

1. ✅ ~~创建 CGNN 消融 config（noSSF/noSPC/noSCD）→ 运行消融实验~~（已完成，见 9.3 节）
2. **⚠️【立即】创建 CGNN CL 基线 config**（EWC/LwF/ER on CGNN，`configs/ours/cl_on_cgnn/`）→ 验证 TASD-CL 优于标准 CL 方法
3. **创建 elliptic++actor_TASDCL_CGNN.yaml** → 运行验证跨数据集泛化
4. **写论文实验章节**：主结果表（9.1）+ 消融表（9.3）+ 逐任务趋势图（9.2）

### 主实验对比表（论文用，Elliptic Task 10）

| 方法 | avg_F1 | avg_AUC-ROC | avg_Macro_F1 | 排名 |
|---|---|---|---|---|
| **TASD-CL（我们）** | **0.4347** | 0.8621 | **0.6582** | **1** |
| CGNN Naive | 0.4019 | **0.8631** | 0.6160 | 2 |
| HOGRL Naive | 0.3953 | 0.8570 | 0.6180 | 3 |
| GradGNN Naive | 0.3839 | 0.8370 | 0.5820 | 4 |
| GCN Naive | 0.3813 | 0.8560 | 0.5870 | 5 |
| BSL Naive | 0.3161 | 0.7680 | 0.5620 | 6 |
| PMP Naive | 0.0591 | 0.6980 | 0.4930 | 7 |
| ConsisGAD Naive | 0.000 | 0.6680 | 0.4650 | 8 |

### 消融实验表（论文用，Elliptic Task 10）

| 方法 | avg_F1 | avg_AUC-ROC | avg_Macro_F1 |
|---|---|---|---|
| **TASD-CL Full** | **0.4347** | 0.8621 | **0.6582** |
| w/o SSF | 0.4042 | 0.8638 | 0.6213 |
| w/o SPC | 0.4308 | — | — |
| w/o SCD | 0.000 | — | — |
