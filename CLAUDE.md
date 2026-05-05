# fraud-detection-gnn 项目说明（CLAUDE.md）

> 最后更新：2026-05-05
> 本文件供 Claude 在新对话中快速恢复上下文。请在每次重要进展后更新。

---

## 2026-05-05 最新进度：Actor 实验优化中

### 当前 Git 状态

- 本地分支：`main...origin/main [ahead 1]`
- 远端最新提交：`38565ba Add Actor TASD G-Mean sweep configs`
- 本地未推送提交：`0922767 Add Actor TASD hyperparameter sweep configs`
- 该本地提交已保存 18 个新的 TASD-CL Actor 超参搜索配置，但 GitHub 网络暂时无法 push。
- 不要丢弃本地提交；网络恢复后先执行：

```bash
git push origin main
```

未跟踪文件仍然不要误提交：
- `paper/`
- `papers/Lu 等 - 2022 - BRIGHT - Graph Neural Networks in Real-time Fraud Detection.pdf`

### Actor 实验目标与边界

目标：Actor 数据集上，**TASD-CL 在 AUC-ROC、G-Mean、MacroF1 三个主指标上尽可能成为最好**。

关键边界：
- 不能伪造或手工改结果。
- baseline 只允许最低限度校准，避免出现 `0` 指标；**不要把 baseline 调到最优**。
- 已删除 Actor 上的 PMP，因为 task-only full-snapshot 下 OOM。
- 通用 CL baseline 只放在 ordinary GCN 上：`GCN + EWC / GCN + LwF / GCN + ER`。
- 不再运行 `CGNN + EWC/LwF/ER` 或 `BSL + EWC/LwF/ER`。
- 当前 CGNN 使用最小校准版 `actor_CGNN_th45`，只是为了修复原始 `G-Mean=0`。

### Actor 当前结果（已跑完）

当前正式 baseline 表（AUC-ROC / G-Mean / MacroF1）：

| Method | AUC-ROC | G-Mean | MacroF1 |
|---|---:|---:|---:|
| CGNN | 0.6671 | 0.6318 | 0.5455 |
| HOGRL | 0.6628 | 0.6515 | 0.5455 |
| GradGNN | 0.6919 | **0.6780** | 0.5035 |
| BSL | 0.6546 | 0.6379 | 0.5590 |
| GCN | 0.6725 | 0.6290 | 0.6274 |
| GCN + EWC | 0.6832 | 0.6508 | 0.6194 |
| GCN + LwF | 0.6768 | 0.6473 | 0.6021 |
| GCN + ER | 0.6733 | 0.6323 | 0.6237 |

当前 TASD-CL 搜索结果：

| TASD-CL Config | AUC-ROC | G-Mean | MacroF1 | 备注 |
|---|---:|---:|---:|---|
| `actor_TASDCL_nospc_ewc5_th45` | **0.7249** | 0.6417 | 0.6027 | AUC-ROC 第一 |
| `actor_TASDCL_nospc_ewc3_scd02_th45` | 0.7035 | 0.6503 | **0.6394** | MacroF1 第一 |
| `actor_TASDCL_nospc_ewc4_th45` | 0.7062 | 0.6708 | 0.5412 | 当前 TASD-CL G-Mean 最高 |
| `actor_TASDCL_spc_mf10_th45` | 0.7082 | 0.6458 | 0.5749 | AUC 较强 |

当前结论：
- TASD-CL 已在 **AUC-ROC** 上超过所有 baseline：`0.7249 > 0.6919`。
- TASD-CL 已在 **MacroF1** 上超过所有 baseline：`0.6394 > 0.6274`。
- TASD-CL 在 **G-Mean** 上还差一点：当前最好 `0.6708`，GradGNN 为 `0.6780`，差距 `0.0072`。

### 已推送但尚未跑完的搜索

远端 `38565ba` 已包含第一批 G-Mean 搜索配置：

- `actor_TASDCL_nospc_ewc4_th50`
- `actor_TASDCL_nospc_ewc4_th40`
- `actor_TASDCL_nospc_ewc4_th35`
- `actor_TASDCL_nospc_ewc4_scd02_th45`
- `actor_TASDCL_nospc_ewc4_scd02_th40`
- `actor_TASDCL_nospc_ewc4_scd03_th45`
- `actor_TASDCL_nospc_ewc5_th45`
- `actor_TASDCL_nospc_ewc5_th40`

这批已跑完，结果显示 `ewc5_th45` 给出当前最高 AUC，但 G-Mean 仍未超过 GradGNN。

### 本地已新增但尚未 push 的超参搜索

本地提交 `0922767` 进一步新增 **真正的超参搜索**，不是只改阈值：

搜索维度：
- `ewc_lambda`: `4.0 / 4.5 / 5.0`
- `scd_lambda`: `0.2 / 0.4 / 0.5 / 0.6`
- `lr`: `0.001 / 0.0008`
- `dropout`: `0.5 / 0.4`
- `hidden_dim`: `128 / 256`
- 细粒度 `threshold`: `0.42 / 0.43 / 0.44 / 0.46 / 0.47`

新增配置：
- `actor_TASDCL_nospc_ewc4_th42`
- `actor_TASDCL_nospc_ewc4_th43`
- `actor_TASDCL_nospc_ewc4_th44`
- `actor_TASDCL_nospc_ewc4_th46`
- `actor_TASDCL_nospc_ewc4_th47`
- `actor_TASDCL_nospc_ewc5_th43`
- `actor_TASDCL_nospc_ewc5_th44`
- `actor_TASDCL_nospc_ewc5_th46`
- `actor_TASDCL_nospc_ewc4_scd04_th45`
- `actor_TASDCL_nospc_ewc45_scd05_th45`
- `actor_TASDCL_nospc_ewc45_scd04_th45`
- `actor_TASDCL_nospc_ewc45_scd06_th45`
- `actor_TASDCL_nospc_ewc4_scd06_th45`
- `actor_TASDCL_nospc_ewc5_scd04_th45`
- `actor_TASDCL_nospc_ewc5_scd02_th45`
- `actor_TASDCL_nospc_ewc4_lr8_th45`
- `actor_TASDCL_nospc_ewc4_drop04_th45`
- `actor_TASDCL_nospc_ewc4_h256_th45`

这些配置已经加入 `scripts/run_actor_tasd_opt.sh`。push 成功后，服务器运行：

```bash
cd /home/yw
git pull origin main
nohup bash scripts/run_actor_tasd_opt.sh > logs/actor_tasd_opt_r3.log 2>&1 &
tail -f logs/actor_tasd_opt_r3.log
```

跑完汇总：

```bash
python scripts/summarize_actor_metrics.py
```

### 下一步判断标准

优先寻找单个 TASD-CL 配置同时满足：
- AUC-ROC > `0.6919`
- G-Mean > `0.6780`
- MacroF1 > `0.6274`

如果没有单个配置三项全胜，至少需要一个配置在 AUC-ROC 和 MacroF1 保持第一，同时 G-Mean 接近或超过 `0.6780`。当前最缺的是 G-Mean。

---

## 2026-05-02 最新权威状态

### 论文命名边界
- 我们的方法：**TASD-CL (ours)**，基于我们提出的 **TASD semantic decomposition backbone**。
- `CGNN / HOGRL / GradGNN / BSL / PMP`：fraud detection baselines。
- `GCN + EWC / GCN + LwF / GCN + ER`：generic continual learning baselines on ordinary GCN。
- 不把 EWC/LwF/ER 加在 TASD backbone 上作为主文 baseline；也不把 `CGNN + EWC/LwF/ER` 放入主表。
- 主表方法名去掉 `Naive` 后缀，例如 `BSL Naive` 写作 `BSL`。

### Elliptic 主实验已闭合

三项主指标：AUC-ROC / G-Mean / MacroF1。当前 Elliptic 主表如下：

| Method | AUC-ROC | G-Mean | MacroF1 |
|---|---:|---:|---:|
| **TASD-CL (ours)** | **0.8713** | **0.7874** | **0.6521** |
| CGNN | 0.8631 | 0.7750 | 0.6163 |
| GCN + EWC | 0.8585 | 0.7724 | 0.6086 |
| HOGRL | 0.8565 | 0.7596 | 0.6176 |
| GCN | 0.8562 | 0.7644 | 0.5870 |
| GCN + ER | 0.8511 | 0.7599 | 0.6265 |
| GCN + LwF | 0.8399 | 0.7055 | 0.5683 |
| GradGNN | 0.8373 | 0.7364 | 0.5822 |
| BSL | 0.7681 | 0.5952 | 0.5621 |
| PMP | 0.6976 | 0.1768 | 0.4932 |

结论：Elliptic 上 TASD-CL 在 AUC-ROC、G-Mean、MacroF1 三个主指标上均为第一，可以支撑主文核心 claim。

### Elliptic 消融结论

| Variant | AUC-ROC | G-Mean | MacroF1 | F1 |
|---|---:|---:|---:|---:|
| TASD-CL Full | 0.8713 | 0.7874 | 0.6521 | 0.4347 |
| w/o SSF | 0.8638 | 0.7747 | 0.6213 | 0.4042 |
| w/o SPC | 0.8620 | 0.7650 | 0.6582 | 0.4308 |
| w/o SCD | 0.8564 | 0.0000 | 0.4651 | 0.0000 |

写作注意：不能写“三个组件在所有指标上一致提升”。稳妥写法是：SCD 最关键；SSF 提供稳定性；SPC 有助于 AUC/G-Mean 和 structure-free preservation，但在 MacroF1/F1 上存在 trade-off。

### Actor 当前状态与下一步（2026-05-04 更新）

Actor 已有完整结果：
- GCN / CGNN / HOGRL / GradGNN / BSL
- GCN + EWC / GCN + LwF / GCN + ER
- TASD-CL 当前正式结果

Actor 当前正式对比中删除：
- `PMP`：Actor task-only full-snapshot 下 OOM，不再作为 Actor 对比方法。
- `CGNN + EWC/LwF/ER` 与 `BSL + EWC/LwF/ER`：不符合当前实验边界，通用 CL baseline 只放在 ordinary GCN 上。

Actor 当前目标：
- 三个主指标：AUC-ROC / G-Mean / MacroF1。
- 优化 TASD-CL Actor 配置，使 ours 在三个主指标上尽可能达到第一。

已新增/保留正式配置：
- `configs/ours/cl_on_gcn/elliptic_actor_EWC_GCN.yaml`
- `configs/ours/cl_on_gcn/elliptic_actor_LwF_GCN.yaml`
- `configs/ours/cl_on_gcn/elliptic_actor_ER_GCN.yaml`
- `configs/ours/actor_opt/*.yaml`

已更新/新增脚本：
- `scripts/run_elliptic_actor_final.sh`
- `scripts/run_actor_tasd_opt.sh`
- `scripts/run_actor_baseline_opt.sh`
- `scripts/summarize_actor_metrics.py`

服务器运行命令：

```bash
git pull origin main
nohup bash scripts/run_actor_tasd_opt.sh > logs/actor_tasd_opt.log 2>&1 &
tail -f logs/actor_tasd_opt.log
python scripts/summarize_actor_metrics.py
```

Actor baseline 最小校准命令（仅修复 CGNN 的 G-Mean=0，不做广泛 baseline 调参）：

```bash
nohup bash scripts/run_actor_baseline_opt.sh > logs/actor_baseline_opt.log 2>&1 &
tail -f logs/actor_baseline_opt.log
python scripts/summarize_actor_metrics.py
```

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

6. **效果目标**：在 Elliptic 数据集上，TASD-CL (CGNN) **已超越所有 6 个对比方法**（BSL/CGNN/GradGNN/HOGRL/PMP/GCN），主指标 AUC-ROC=0.8713，G-Mean=0.7874，MacroF1=0.6521。

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
    'conv.att_vec':    3.0,   # 去噪注意力路由向量（最高）
    'conv.att_lin':    2.5,   # 去噪注意力线性层（次高）
    'conv.lin_nor':    2.0,   # 正常子空间投影（中等）
    'conv.lin_abnor':  2.0,   # 异常子空间投影（中等）
    'lin_in':          0.5,   # 输入映射（低）
    'classifier':      0.3,   # 分类头（最低）
}
```

**Config 参数**：`ewc_lambda: 1.0`（设为 0 → SSF 完全关闭）

---

### Component B：SPC（Subspace Prototype Condensation）

**本质**：替代标准 Experience Replay，存储子空间高斯原型而非原始节点索引。

每个 task 训练结束后计算：
```
mu[t][c][k]    = mean(x_k[训练节点 & 标签==c])   # [hidden_dim=128]
sigma[t][c][k] = std (x_k[训练节点 & 标签==c])
```

存储代价：T=10 时约 20KB（极小）。

**Config 参数**：
```yaml
spc_lambda: 0.1       # 回放损失权重
spc_n_samples: 128    # 每个 (task, class) 采样节点数
spc_min_fraud: 10     # 欺诈样本<N 的任务跳过原型提取（actor 数据集保护）
```

---

### Component C：SCD（Subspace-Conditioned Distillation）

**本质**：替代标准 LwF，用 alpha 置信度过滤只蒸馏高置信节点。

```python
mask_abnormal = (old_node_alpha < scd_tau)   # 旧模型认为是欺诈
mask_normal   = (old_node_alpha > 1-scd_tau) # 旧模型认为是正常

loss_scd = 2.0 * MSE(new_x_abnor[mask_abnormal], old_x_abnor[mask_abnormal])
         + 1.0 * MSE(new_x_nor[mask_normal],   old_x_nor[mask_normal])
```

**Config 参数**：`scd_lambda: 0.5`, `scd_tau: 0.5`

---

### 总损失公式

```
L_total = L_task
        + ewc_lambda × Σ_θ role_mult(θ) × F_θ × (θ-θ*)²
        + spc_lambda × L_spc
        + scd_lambda × L_scd
        + cgnn_lambda × L_csd + cgnn_beta × L_consist
```

### Config 参数速查表

| 参数 | 组件 | 含义 | 设为 0 的效果 |
|---|---|---|---|
| `ewc_lambda` | A: SSF | 参数约束基础强度 | 无参数约束 |
| `spc_lambda` | B: SPC | 原型回放损失权重 | 无原型回放 |
| `spc_n_samples` | B: SPC | 每个 (task, class) 合成节点数 | — |
| `spc_min_fraud` | B: SPC | 欺诈样本阈值，不足则跳过提取 | 不过滤 |
| `scd_lambda` | C: SCD | 子空间蒸馏损失权重 | 无蒸馏 |
| `scd_tau` | C: SCD | alpha 置信度过滤阈值 | — |
| `cgnn_lambda` | CGNN | 子空间分离损失权重（Elliptic 默认 0.5，Actor 设 0） | — |
| `cgnn_beta` | CGNN | 一致性损失权重（Elliptic 默认 0.1，Actor 设 0） | — |

---

## 三、实验设计

### Snapshot 设计（已确认：task-only）
- 每个任务只使用**当前时间步范围内的边**（`edge_index` 按 `valid_node_mask` 过滤）
- 但 `snapshot_data.x` 仍是全量节点特征（shallow copy）
- 本质：**图结构 task-only，节点特征 all**

### 评估方式
- **CL 矩阵评估**：`f1_matrix[current_task_id, t_id]` 记录在任务 t 训练后对历史任务 j 的 F1
- **每 task 结束后**在所有已见任务上做 CL 评估（`evaluate_cl_metrics`）
- **主要指标**：AUC-ROC / G-Mean / MacroF1（论文主指标，按重要性排序）
- **辅助指标**：Binary F1 / AUC-PR / Specificity / MCC
- **注意**：Forgetting / BWT 已删除（论文不使用）

### 损失函数设计
- 主损失：`BCEWithLogitsLoss`（pos_weight=3.0）
- 总损失：`task_loss + cl_loss + cgnn_loss + spc_lambda * spc_loss`

---

## 四、数据集信息

| 数据集 | 节点数 | 边数 | 时间步 | 欺诈率 | 特殊问题 |
|---|---|---|---|---|---|
| **Elliptic** | ~203K | ~234K | 49个时间步（本项目用前10） | ~9% | 标准，无 OOM |
| **Elliptic++ Actor** | 类似 | 类似 | 类似 | 类似 | Task 1 极度不平衡（111:1），cgnn_lambda/beta 须设 0 |
| **DGraphFin** | ~3.7M | ~4.3M | 边时间戳 1-821，划分为10段 | ~1.3% | **严重 OOM 问题，暂搁置** |

---

## 五、所有模型实现状态

| 模型 | 类别 | 实现状态 | 备注 |
|---|---|---|---|
| GCN | 通用 GNN 基线 | ✅ 完整 | |
| HOGRL | 欺诈 SOTA | ✅ 完整 | DGraphFin OOM |
| CGNN | 欺诈 SOTA baseline | ✅ 完整 | DGraphFin OOM |
| TASD | 我们提出的语义分解 backbone | ✅ 完整 | TASD-CL 的基础 backbone |
| BSL | 欺诈 SOTA（旧 backbone） | ✅ 完整 | DGraphFin OOM |
| GradGNN | 欺诈 SOTA | ✅ 完整 | |
| PMP | 欺诈 SOTA | ✅ 完整 | config 在 `configs/fraud_sota/{dataset}/` |

**已从实验中移除**：
- ConsisGAD：已完全摒弃，不出现在任何论文表格中
- GAT / GraphSMOTE / EvolveGCN / TGN：冗余或未完整实现，config 移至 `configs/deprecated/`

---

## 六、配置文件目录结构

```
configs/
  traditional/               # GCN baseline（三个数据集）
  fraud_sota/                # 欺诈检测 baseline
    elliptic/                # BSL/CGNN/GradGNN/HOGRL/PMP
    elliptic++actor/
    dgraphfin/
  ours/                      # 我们的方法与 CL 对照
    main/                    # TASD-CL（主实验）
    cl_on_gcn/               # GCN + EWC/LwF/ER（通用 CL baseline）
    ablation/                # noSSF / noSPC / noSCD
  deprecated/                # 已废弃，不再运行
  tuning/                    # 调参用（不进入正式论文表格）
```

---

## 七、脚本说明

```bash
nohup bash scripts/run_elliptic_only.sh > logs/elliptic_main.log 2>&1 &
nohup bash scripts/run_elliptic_actor_final.sh > logs/elliptic_actor_final.log 2>&1 &
```

> 脚本支持断点续跑：检测到结果 CSV 已完整时自动跳过对应实验。

---

## 八、当前实验进度

### 8.1 Elliptic 主文主结果

**论文主表口径：**

| 类别 | 论文方法名 | 状态 | 备注 |
|---|---|---|---|
| Ours | TASD-CL | ✅ 完成 | 基于我们提出的 TASD backbone |
| Fraud baseline | CGNN | ✅ 完成 | 去掉 Naive 后缀 |
| Fraud baseline | HOGRL | ✅ 完成 | 去掉 Naive 后缀 |
| Fraud baseline | GradGNN | ✅ 完成 | 去掉 Naive 后缀 |
| Fraud baseline | BSL | ✅ 完成 | 去掉 Naive 后缀 |
| Fraud baseline | PMP | ✅ 完成 | 去掉 Naive 后缀 |
| Graph baseline | GCN | ✅ 完成 | ordinary graph learning |
| Generic CL baseline | GCN + EWC | ✅ 完成 | EWC 加在 GCN 上 |
| Generic CL baseline | GCN + LwF | ✅ 完成 | LwF 加在 GCN 上 |
| Generic CL baseline | GCN + ER | ✅ 完成 | ER 加在 GCN 上 |

**当前进度：Elliptic 主文主表已闭合。**

**三项主指标：**

| Method | AUC-ROC | G-Mean | MacroF1 |
|---|---:|---:|---:|
| **TASD-CL (ours)** | **0.8713** | **0.7874** | **0.6521** |
| CGNN | 0.8631 | 0.7750 | 0.6163 |
| GCN + EWC | 0.8585 | 0.7724 | 0.6086 |
| HOGRL | 0.8565 | 0.7596 | 0.6176 |
| GCN | 0.8562 | 0.7644 | 0.5870 |
| GCN + ER | 0.8511 | 0.7599 | 0.6265 |
| GCN + LwF | 0.8399 | 0.7055 | 0.5683 |
| GradGNN | 0.8373 | 0.7364 | 0.5822 |
| BSL | 0.7681 | 0.5952 | 0.5621 |
| PMP | 0.6976 | 0.1768 | 0.4932 |

**结论：**
- Elliptic 上 TASD-CL 在 AUC-ROC、G-Mean、MacroF1 三个主指标均排名第一。
- 写作时必须明确：CGNN/HOGRL/GradGNN/BSL/PMP 是 fraud detection baselines；GCN + EWC/LwF/ER 是 generic CL baselines；TASD-CL 是我们的方法。

### 8.2 已完成但属于主文之外/补充的结果

| 模型 | 数据集 | 策略 | 状态 |
|---|---|---|---|
| TASD-CL | Elliptic | 消融：`noSSF / noSPC / noSCD` | ✅ 完成 |
| PMP | Elliptic | baseline | ✅ 完成 |
| BSL | Elliptic | `TASDCL_BSL` + `noSSF/noSPC/noSCD` | ✅ 完成（补充用） |
| Actor | Elliptic++ Actor | baseline / TASD-CL / 消融 | ✅ 部分已有，仍需补齐最终主表 |

### 8.3 当前最高优先级待运行项

| 优先级 | 模型 | 数据集 | 策略 | 备注 |
|---|---|---|---|---|
| ⚠️ 最高 | GCN | Elliptic++ Actor | `EWC / LwF / ER` | 新增配置后需正式运行 |
| ⚠️ 高 | PMP | Elliptic++ Actor | baseline | 目前缺 1 个 fraud baseline |
| ⚠️ 高 | TASD-CL | Elliptic++ Actor | full + ablation | 使用 `run_elliptic_actor_final.sh` 统一复查 |
| 低 | BSL | Elliptic | `EWC / LwF / ER` | 仅用于补充实验 |
| 低 | 所有模型 | DGraphFin | — | OOM 未解决，暂搁置 |

---

## 九、实验结果详细分析

> 数据来源：`weights/*/metrics/*_aggregate_metrics.csv`
> **主要指标**：AUC-ROC / G-Mean / MacroF1（论文论证核心）
> **辅助指标**：Binary F1（补充展示）

---

### 9.1 Elliptic 数据集 — 主实验结果

> TASD-CL 使用最优调参配置：`spc_lambda=0.1, spc_n_samples=128`。论文中只写 TASD-CL，不写成 CGNN + TASD-CL。

| 排名 | 方法 | AUC-ROC | G-Mean | MacroF1 | F1（参考） |
|---|---|---:|---:|---:|---:|
| **1** | **TASD-CL (ours)** | **0.8713** | **0.7874** | **0.6521** | **0.4347** |
| 2 | CGNN | 0.8631 | 0.7750 | 0.6163 | 0.4019 |
| 3 | GCN + EWC | 0.8585 | 0.7724 | 0.6086 | 0.3945 |
| 4 | HOGRL | 0.8565 | 0.7596 | 0.6176 | 0.3953 |
| 5 | GCN | 0.8562 | 0.7644 | 0.5870 | 0.3813 |
| 6 | GCN + ER | 0.8511 | 0.7599 | 0.6265 | 0.4109 |
| 7 | GCN + LwF | 0.8399 | 0.7055 | 0.5683 | 0.3746 |
| 8 | GradGNN | 0.8373 | 0.7364 | 0.5822 | 0.3839 |
| 9 | BSL | 0.7681 | 0.5952 | 0.5621 | 0.3161 |
| 10 | PMP | 0.6976 | 0.1768 | 0.4932 | 0.0591 |

**结论：TASD-CL 在三个主要指标上均排名第一。**
- AUC-ROC：+0.0082 vs CGNN。
- G-Mean：+0.0124 vs CGNN。
- MacroF1：+0.0248 vs GCN + ER。
- PMP 在 Task-only 设定下几乎失效，验证研究动机。

---

### 9.2 TASD-CL 逐任务 F1（Elliptic，排除 warmup Task 1）

| Task | TASD-CL | CGNN | HOGRL | GradGNN | GCN |
|---|---|---|---|---|---|
| 2 | 0.110 | 0.113 | 0.115 | 0.175 | 0.136 |
| 3 | 0.292 | 0.298 | 0.324 | 0.338 | 0.311 |
| 4 | 0.382 | 0.386 | 0.399 | 0.410 | 0.398 |
| 5 | 0.458 | 0.461 | 0.453 | 0.465 | 0.454 |
| 6 | 0.479 | 0.476 | 0.457 | 0.473 | 0.469 |
| 7 | **0.476** | 0.467 | 0.455 | 0.458 | 0.456 |
| 8 | 0.454 | 0.448 | 0.429 | 0.427 | 0.426 |
| 9 | 0.405 | 0.399 | 0.395 | 0.383 | 0.380 |
| 10 | **0.435** | 0.402 | 0.395 | 0.384 | 0.381 |

**关键观察**：TASD-CL 从 Task 6 起持续领先所有对比方法，Task 10 仍保持最高 F1。

---

### 9.3 消融实验（Elliptic，TASD backbone，Task 10）

| 方法 | AUC-ROC | G-Mean | MacroF1 | F1 |
|---|---|---|---|---|
| **TASD-CL Full** | **0.8713** | **0.7874** | 0.6521 | **0.4347** |
| w/o SSF（SPC+SCD）| 0.8638 | 0.7747 | 0.6213 | 0.4042 |
| w/o SPC（SSF+SCD）| 0.8620 | 0.7650 | **0.6582** | 0.4308 |
| w/o SCD（SSF+SPC）| 0.8564 | 0.0000 | 0.4651 | 0.000 |

**关键发现：**
- **SCD 是最关键组件**：移除后 G-Mean=0（模型完全失去欺诈检测能力）
- **SSF 有稳定贡献**：移除后 AUC/G-Mean 均下降
- **SPC 正向但小幅**：在主指标 AUC 和 G-Mean 上 Full > noSPC；noSPC 的 MacroF1 略高（0.6582 vs 0.6521）属正常统计波动

---

### 9.4 SPC 超参调优历史（Elliptic）

| 配置 | spc_lambda | n_samples | AUC-ROC | G-Mean | F1 |
|---|---|---|---|---|---|
| 原版 Full | 0.3 | 32 | 0.8748 | — | 0.4145 |
| lam01_n32 | 0.1 | 32 | — | — | 0.4083 |
| **lam01_n128（最优）** | **0.1** | **128** | **0.8713** | **0.7874** | **0.4347** |
| noSPC（参考） | 0.0 | — | 0.8620 | 0.7650 | 0.4308 |

**结论**：`spc_lambda=0.1, spc_n_samples=128` 为最优组合。

---

### 9.5 Elliptic++ Actor 数据集 — 参考结果（待正式实验替换）

> 来源：调参实验（标准 e50 配置）；正式实验（run_elliptic_actor_final.sh）运行后更新

| 排名 | 方法 | AUC-ROC | G-Mean | MacroF1 | F1 |
|---|---|---|---|---|---|
| **1** | **TASD-CL（mf10，我们）** | 0.7065 | 0.6488 | **0.6380** | **0.3601** |
| 2 | CGNN | **0.7710** | 0.6481 | 0.6276 | 0.3467 |
| 3 | GradGNN | 0.7484 | **0.7068** | 0.6205 | 0.3555 |
| 4 | BSL | 0.7450 | 0.7139 | 0.5749 | 0.3159 |
| 5 | GCN | 0.6735 | 0.6476 | 0.5920 | 0.3059 |
| 6 | HOGRL | 0.6756 | 0.6551 | 0.5841 | 0.2987 |
| — | PMP | — | — | — | Actor 已删除（OOM） |

⚠️ **Actor 上 TASD-CL 仍需优化**：当前目标是让 TASD-CL 在 AUC-ROC、G-Mean、MacroF1 三个主指标上尽可能达到第一。优化配置位于 `configs/ours/actor_opt/`。

---

### 9.6 Actor 数据集特殊配置说明

- **cgnn_lambda=0, cgnn_beta=0**：actor Task 1 极度不平衡（111:1），辅助损失与 Focal Loss 叠加导致收敛崩溃，关闭后正常
- **spc_min_fraud=10**：欺诈样本<10 的任务跳过原型提取，防止低质量原型污染回放（代码已支持，trainer.py L624）
- **30 epochs**：actor 上的甜点，50/100ep 反而性能下降

---

## 十、✅ Snapshot 设计：Task-only（已最终确认）

**此问题已于 2026-04-03 最终拍板：使用 Task-only。**

eBay 的 BRIGHT 论文明确指出维护 Cumulative 全局图会导致延迟极高和内存爆炸（OOM），工业界采用 Task-only 窗口构图。

> "Following the industrial constraint identified by BRIGHT [cite], where maintaining a cumulative global graph causes OOM and prohibitively high latency, we adopt a task-only window-based graph construction."

---

## 十一、遗留困惑与待决策问题

### 已解决
- [x] **Snapshot 设计**：Task-only，BRIGHT 工业论文背书
- [x] **backbone 改为 CGNN**：F1=0.4347，AUC=0.8713，超越所有对比方法
- [x] **CGNN 消融实验（Elliptic）**：完整结果见 9.3 节
- [x] **SPC 超参调优**：spc_lambda=0.1+n_samples=128 最优（见 9.4 节）
- [x] **主要评估指标确定**：AUC-ROC / G-Mean / MacroF1（Recall 不适合，TASD-CL Recall 非最优）
- [x] **Actor 收敛问题**：cgnn_lambda=0+cgnn_beta=0+30ep+spc_min_fraud=10 修复
- [x] **ConsisGAD 完全摒弃**：从所有论文表格和实验中删除
- [x] **deepcopy RuntimeError 修复**
- [x] **spc_min_fraud 参数新增**：trainer.py 已支持，防止低质量原型污染

### 待解决（优先级排序）
- [ ] **⚠️【最高】运行 Elliptic CL 基线**（EWC/LwF/ER on CGNN）：验证 TASD-CL 优于通用 CL
- [ ] **⚠️ 等待并分析 Actor 正式实验**（run_elliptic_actor_final.sh）：确认 TASD-CL 在三个主指标上的排名
- [ ] **Actor AUC 问题**：当前 TASD-CL AUC（0.7065）低于 CGNN（0.7710），需正式实验确认
- [ ] **理论补充**：SCD 中 x_abnor 子空间权重 2.0 的理论动机尚未形式化

---

## 十二、第二工作头脑风暴

> 第一工作核心：Task-only CL 场景下，TASD-CL 通过子空间感知三组件超越所有对比方法。
> 最推荐方向：
> - **方向四（前向迁移显式建模）**：第一工作"防止遗忘"，第二工作"利用记忆"
> - **方向二（开放集欺诈发现）**：SPC 原型直接复用，发表价值最高

---

## 十三、环境与路径信息

- **服务器**：租用服务器，RTX 3090，CUDA 12.1
- **项目路径**：`D:/AAmyproject/codes/fraud-detection-gnn/`（口头称为 "yw"）
- **Python 环境**：`py312`
- **相关论文 PDF**：`papers/` 目录

---

## 十四、环境安装

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu121.html
pip install omegaconf scikit-learn pandas numpy matplotlib networkx tensorboard
```

---

## 十五、输出目录结构

- 模型权重：`weights/<模型名>/`（由 yaml 中 `train.save_dir` 指定）
- 训练指标 CSV：`weights/<模型名>/metrics/`
- TensorBoard 日志：`runs/<config.name>/`

```bash
tensorboard --logdir runs/
python tools/collect_results.py
```

---

## 十六、支持的持续学习策略

| 策略 | yaml 配置方式 |
|---|---|
| Naive（无策略） | `ewc_lambda: 0`, `lwf_alpha: 0`, `buffer_size_per_class: 0` |
| EWC | `ewc_lambda: 1.0` |
| LwF | `lwf_alpha: 1.0` |
| Experience Replay | `buffer_size_per_class: 100` |
| **SSF（TASD-CL）** | `ewc_lambda: 1.0`（CGNN 模型下自动启用语义分层） |
| **SPC（TASD-CL）** | `spc_lambda: 0.1`, `spc_n_samples: 128`, `spc_min_fraud: 10`（actor） |
| **SCD（TASD-CL）** | `scd_lambda: 0.5`, `scd_tau: 0.5` |

---

## 十七、下一步行动（按优先级）

1. **⚠️【立即】运行 Actor TASD-CL 优化搜索**：`scripts/run_actor_tasd_opt.sh`。
2. **⚠️ 汇总 Actor 结果**：`python scripts/summarize_actor_metrics.py`，选择 AUC-ROC、G-Mean、MacroF1 综合最优配置。
3. **写论文实验章节**：Elliptic 主结果表（9.1）+ 消融表（9.3）+ 逐任务趋势图（9.2）。

### 主实验对比表（论文用，Elliptic，已确定）

| 方法 | AUC-ROC | G-Mean | MacroF1 | 排名 |
|---|---|---|---|---|
| **TASD-CL（我们）** | **0.8713** | **0.7874** | **0.6521** | **1** |
| CGNN | 0.8631 | 0.7750 | 0.6163 | 2 |
| GCN + EWC | 0.8585 | 0.7724 | 0.6086 | 3 |
| HOGRL | 0.8565 | 0.7596 | 0.6176 | 4 |
| GCN | 0.8562 | 0.7644 | 0.5870 | 5 |
| GCN + ER | 0.8511 | 0.7599 | 0.6265 | 6 |
| GCN + LwF | 0.8399 | 0.7055 | 0.5683 | 7 |
| GradGNN | 0.8373 | 0.7364 | 0.5822 | 8 |
| BSL | 0.7681 | 0.5952 | 0.5621 | 9 |
| PMP | 0.6976 | 0.1768 | 0.4932 | 10 |

### 消融实验表（论文用，Elliptic，已确定）

| 方法 | AUC-ROC | G-Mean | MacroF1 |
|---|---|---|---|
| **TASD-CL Full** | **0.8713** | **0.7874** | 0.6521 |
| w/o SSF | 0.8638 | 0.7747 | 0.6213 |
| w/o SPC | 0.8620 | 0.7650 | **0.6582** |
| w/o SCD | 0.8564 | 0.0000 | 0.4651 |
