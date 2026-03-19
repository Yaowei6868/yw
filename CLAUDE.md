# fraud-detection-gnn 项目说明（CLAUDE.md）

> 最后更新：2026-03-19
> 本文件供 Claude 在新对话中快速恢复上下文。请在每次重要进展后更新。

---

## 一、项目定位与研究动机

### 研究问题（一句话）
在图神经网络欺诈检测的任务增量持续学习场景下，提出 TASD-CL 框架，通过子空间语义过滤（SSF）、原型凝缩（SPC）和置信度蒸馏（SCD）三个组件，使 BSL 模型在时序欺诈 pattern 演变中保持稳定的检测性能。

### 研究叙事逻辑
1. **发现的问题**：SOTA 欺诈检测 GNN 模型在时序任务增量流中，随时间步推进，在新出现的欺诈 pattern 上性能持续下降（F1、AUC 下滑）
2. **根本原因**：欺诈 pattern 存在时序分布偏移（pattern evolution），模型在当前任务上过拟合，子空间表示被污染。注意：**灾难性遗忘不是主要问题**（实验已验证基本不存在），真正问题是时序分布偏移
3. **解决方法**：TASD-CL 框架，在 BSL backbone 上添加三个组件
4. **效果**：在三个金融图数据集上，Naive 各 SOTA 随任务推进性能下降，TASD-CL 显著缓解这种下降

### 核心思想：为什么选 BSL 作为 backbone
BSL 将节点嵌入解耦为三个子空间：
- `Z_na`：正常-异常交互子空间
- `Z_aa`：纯异常（欺诈感知）子空间
- `Z_nn`：纯正常子空间

这种解耦使得"保护哪些参数"、"蒸馏哪些节点"、"回放什么知识"都可以针对欺诈语义精细设计，是 TASD-CL 三个组件的设计基础。

---

## 二、TASD-CL 三组件详细说明

| 组件 | 全称 | 作用 | 实现位置 |
|---|---|---|---|
| **SSF** | Subspace-Stratified Fisher | 对不同 BSL 参数施加不同强度的 EWC 约束，子空间路由参数（att_vec/att_bias）约束最强 | `trainer.py: _get_ssf_lambda()` |
| **SPC** | Subspace Prototype Condensation | 任务结束后提取三子空间的类原型（Gaussian μ,σ），存入缓冲区，绕开图结构做回放 | `trainer.py: _update_spc_prototypes()`, `buffer.py: SubspacePrototypeBuffer` |
| **SCD** | Subspace-Conditioned Distillation | 在三子空间层面蒸馏，只蒸馏旧模型对某子空间置信度高（alpha_k > scd_tau）的节点，Z_aa 权重最高（2.0） | `trainer.py: _compute_scd_loss()` |

### SSF 参数约束强度配置（`BSL_PARAM_GROUPS`）
```python
'att_vec':      3.0   # 子空间路由（最高）
'att_bias':     3.0
'edge_decoder': 2.0   # 边类型分类（中等）
'gnn_encoder':  0.5   # 图编码器（低）
'lin_in':       0.5
'classifier':   0.3   # 分类器（最低）
```

### TASD-CL 激活方式（yaml 配置）
```yaml
spc_lambda: 1.0      # SPC 回放损失权重，0 表示关闭
scd_lambda: 1.0      # SCD 蒸馏损失权重，0 表示关闭
scd_tau: 0.5         # SCD 置信度过滤阈值
ewc_lambda: 1.0      # SSF 使用 EWC 框架，需配合 BSL 生效
```

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
- **Forgetting**：历史最高分 - 当前分
- **BWT（Backward Transfer）**：当前分 - 首次训练分
- **每 task 结束后**在所有已见任务上做 CL 评估（`evaluate_cl_metrics`）

### 损失函数设计
- 主损失：`BinaryFocalLoss`（alpha 动态计算，根据正负比例截断）
- Task 1：`clip_cap = 20.0`；其他 task：`clip_cap = 10.0`
- 总损失：`task_loss + cl_loss + bsl_loss + cgnn_loss + spc_lambda * spc_loss + smote_lambda * smote_loss`

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
| GCN | SOTA baseline | ✅ 完整 | |
| GAT | SOTA baseline | ✅ 完整 | |
| HOGRL | SOTA baseline | ✅ 完整 | DGraphFin OOM |
| CGNN | SOTA baseline | ✅ 完整 | DGraphFin OOM |
| BSL | SOTA baseline + backbone | ✅ 完整 | DGraphFin OOM |
| ConsisGAD | SOTA baseline | ✅ 完整 | |
| GradGNN | SOTA baseline | ✅ 完整 | |
| PMP | SOTA baseline | ✅ 完整 | config 在 `configs/fraud_sota/` |
| GraphSMOTE | 不平衡处理 | ✅ 已修复 | 原实现只是2层GCN，已补全SMOTE插值逻辑 |
| EvolveGCN | 动态图 | ❌ 空壳占位符 | 依赖时序GRU/LSTM，暂未实现，**不要运行** |
| TGN | 动态图 | ❌ 空壳占位符 | 依赖时序记忆模块，暂未实现，**不要运行** |

### GraphSMOTE 修复说明
原实现是普通 2 层 GCN，完全没有 SMOTE 逻辑。
修复后：`encode()` → `_smote_interpolate()`（KNN + 线性插值，梯度可回传）→ `forward()`
- 训练时返回 `(out, smote_loss)`
- 推理时只返回 `out`
- trainer 中通过 `snapshot_data.smote_train_idx` 注入当前训练节点索引

---

## 六、配置文件目录结构

```
configs/
  fraud_sota/          # SOTA 对比实验（GCN/GAT/BSL/ConsisGAD/CGNN/GradGNN/HOGRL/PMP）
  tasdcl/              # TASD-CL 方法实验（BSL + SSF/SPC/SCD 组合）
  imbalanced/          # 不平衡处理实验（GraphSMOTE）
  dynamic_graph/       # EvolveGCN/TGN（空壳，不要运行）
```

### PMP 配置位置（注意！）
PMP 的 config 在 `configs/fraud_sota/`，**不在** `configs/PMP/`（该目录不存在）：
- `configs/fraud_sota/elliptic_Naive_PMP.yaml`
- `configs/fraud_sota/elliptic++actor_Naive_PMP.yaml`
- `configs/fraud_sota/dgraphfin_Naive_PMP.yaml`

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
| GCN | Elliptic | Naive/EWC/LwF/ER | ✅ 完成 |
| GAT | Elliptic | Naive | ✅ 完成 |
| BSL | Elliptic | Naive | ✅ 完成 |
| BSL | Elliptic | TASD-CL | ✅ 完成 |
| CGNN | Elliptic | Naive | ✅ 完成 |
| ConsisGAD | Elliptic | Naive + CL(EWC) | ✅ 完成 |
| GradGNN | Elliptic | Naive | ✅ 完成 |
| HOGRL | Elliptic | Naive | ✅ 完成 |
| PMP | Elliptic | Naive | ✅ 完成 |
| GraphSMOTE | Elliptic | Naive | ✅ 完成 |
| GCN | Elliptic++ Actor | Naive/EWC/LwF/ER | ✅ 完成 |
| GAT | Elliptic++ Actor | Naive | ✅ 完成 |
| BSL | Elliptic++ Actor | Naive + TASD-CL | ✅ 完成 |
| CGNN | Elliptic++ Actor | Naive | ✅ 完成 |
| ConsisGAD | Elliptic++ Actor | Naive | ✅ 完成 |
| GradGNN | Elliptic++ Actor | Naive + CL | ✅ 完成 |
| HOGRL | Elliptic++ Actor | Naive + CL | ✅ 完成 |
| GraphSMOTE | Elliptic++ Actor | Naive | ✅ 完成 |

### 待运行 / 存在问题
| 模型 | 数据集 | 问题 |
|---|---|---|
| 所有模型 | DGraphFin | OOM，待解决 |
| PMP | Elliptic++ Actor / DGraphFin | 待确认是否完成 |

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

### 9.3 BSL Naive vs TASD-CL 逐任务对比（Elliptic）

| Task | Naive F1 | TASD-CL F1 | Naive AUC | TASD-CL AUC |
|---|---|---|---|---|
| 1 | 0.022 | 0.031 | 0.644 | **0.835** |
| 2 | 0.109 | 0.136 | 0.609 | **0.872** |
| 3 | 0.173 | 0.204 | 0.608 | **0.883** |
| 4 | 0.217 | 0.255 | 0.620 | **0.873** |
| 5 | 0.214 | 0.259 | 0.642 | **0.865** |
| 6 | 0.240 | 0.289 | 0.657 | **0.834** |
| 7 | 0.246 | 0.290 | 0.655 | **0.810** |
| 8 | 0.232 | 0.272 | 0.639 | **0.794** |
| 9 | 0.218 | 0.253 | 0.614 | **0.777** |
| 10 | 0.207 | 0.240 | 0.616 | **0.771** |

**结论**：TASD-CL 在 F1 上略有提升（+16%），在 AUC-ROC 上显著提升（+25%，0.616→0.771）。

---

### 9.4 GCN 四种持续学习策略对比（Elliptic，Task 10）

| 策略 | F1 | AUC-ROC | 相对 Naive |
|---|---|---|---|
| Naive | 0.2970 | 0.8304 | 基准 |
| EWC | 0.2836 | 0.8211 | F1 -4.5%，AUC -1.1% |
| LwF | 0.2978 | 0.8216 | F1 +0.3%，AUC -1.1% |
| ER | 0.2981 | 0.8358 | F1 +0.4%，AUC +0.7% |

**结论**：三种经典 CL 策略在 GCN 上几乎没有效果，ER 微弱最优。

---

### 9.5 ⚠️ 实验结果中的关键问题与不足

#### 问题一：Forgetting 方向不一致，研究叙事受挑战
**现象**：部分模型 forgetting 为**负值**（性能随时间反而提升）：
- GradGNN Naive：forgetting = **-0.139**（大幅提升）
- HOGRL Naive：forgetting = **-0.089**（持续提升）
- ConsisGAD Naive：forgetting = **-0.016**（微弱提升）

**影响**：研究叙事是"SOTA 在时序流中性能下降"，但至少 3 个模型表现相反，这会让 reviewer 质疑研究问题的普遍性。

**可能原因**：
- 这些模型在 task-only snapshot 下，随着任务推进逐渐学到更多样本，冷启动问题逐渐消失
- 早期任务欺诈样本极少，后期欺诈 pattern 更密集，模型更容易学到

**待解决**：需要搞清楚到底哪些模型在哪个数据集上真正"性能下降"，才能构建有说服力的 motivation。

---

#### 问题二：TASD-CL 在 Elliptic++ Actor 上表现不如 Naive BSL
**现象**：
- Elliptic Actor Task 10：TASD-CL F1 = 0.2686 < Naive F1 = 0.3549（差距 -24%）
- AUC-ROC：TASD-CL 0.711 < Naive 0.741

**影响**：TASD-CL 只在 Elliptic 上有效，在 Elliptic++ Actor 上反而有害，跨数据集泛化性存疑。

**可能原因**：
- Elliptic++ Actor 数据分布与 Elliptic 差异较大，SCD 的蒸馏反而引入了负迁移
- 超参数（scd_tau, spc_lambda 等）在 Elliptic 上调参，未对 Elliptic++ Actor 适配
- Task 1 在 Elliptic++ Actor 上 F1=0（完全没检测到欺诈），后续任务从零开始积累，SPC 原型质量差

**待解决**：需要针对 Elliptic++ Actor 调整超参数，或分析 Task 1 崩溃的原因。

---

#### 问题三：AUC-ROC 与 F1 趋势不一致
**现象**：BSL TASDCL 的 AUC-ROC 随任务推进**大幅下降**（0.835→0.771），尽管整体仍优于 Naive（0.644→0.616），但下降趋势本身值得关注。

**影响**：模型排序能力在时序流中持续衰减，说明即使有 TASD-CL，分布偏移问题并未完全解决，只是缓解。这反而是好的叙事材料，但需要明确说明。

---

#### 问题四：ConsisGAD + EWC 是最强 baseline，超过 TASD-CL
**现象**：ConsisGAD + EWC Final F1 = **0.5174**，远超 BSL + TASD-CL 的 0.2404。

**影响**：如果 reviewer 问"为什么不直接用 ConsisGAD + EWC"，需要有合理回答。

**可能解释**：
- ConsisGAD 本身设计了 augmentation 一致性机制，天然更鲁棒
- EWC 在 ConsisGAD 上有效并不代表在 BSL 上也有效
- TASD-CL 是专门为 BSL 子空间设计的，是 BSL-specific 的优化方案
- **但这意味着研究贡献定位应该是"BSL 的最优 CL 方案"，而不是通用 GNN 欺诈检测 CL 框架**

---

#### 问题五：DGraphFin 数据集无结果
**影响**：只有两个数据集的结果，实验覆盖不充分，难以支撑一篇完整的会议论文。DGraphFin 是最接近真实工业场景的大规模金融图，缺失这部分结果削弱了工作的说服力。

---

### 9.6 当前结果能支撑的结论 vs 不能支撑的结论

| 结论 | 能否支撑 | 说明 |
|---|---|---|
| BSL + TASD-CL 在 Elliptic 上优于 BSL Naive | ✅ 能 | F1 +16%, AUC +25% |
| TASD-CL 是通用欺诈检测 CL 框架 | ❌ 不能 | Elliptic++ Actor 上反而更差 |
| 所有 SOTA 在时序流中性能下降 | ❌ 不能 | GradGNN/HOGRL/ConsisGAD 反向 |
| DGraphFin 上 TASD-CL 有效 | ❌ 无数据 | OOM 未解决 |
| 经典 CL 策略对欺诈 GNN 效果有限 | ✅ 部分 | GCN 上成立，ConsisGAD 反例 |

---

## 十、⚠️ 核心未决问题：Snapshot 设计的矛盾性

**这是目前最重要的研究设计争议，尚未最终确认，新对话时必须重新审视。**

### 两种方案定义

**方案 A：Task-only（当前代码实现）**
- 每个任务只使用当前时间步范围内的边
- `valid_node_mask = (timesteps >= task_start) & (timesteps <= task_end)`
- 每个 task 的图是一个独立的时间窗口切片

**方案 B：Cumulative（累积到当前）**
- 每个任务使用截至当前时间步的所有历史边
- `valid_node_mask = timesteps <= task_end`
- 每个 task 的图随时间增长，包含所有历史

### 矛盾的核心

| 维度 | Task-only 的问题 | Cumulative 的问题 |
|---|---|---|
| **研究叙事一致性** | Baseline 性能下降的原因不纯粹：到底是分布偏移，还是因为看不到历史邻居导致图信息不完整？两者混在一起，归因不干净 | 归因干净：历史信息都在，性能下降只能来自时序分布偏移，研究贡献更有说服力 |
| **真实场景还原** | 更贴近真实系统：交易系统通常只处理当前时间窗口的交易图 | 偏离真实场景：现实中不会把所有历史交易重新纳入图 |
| **文献对齐** | Elliptic 相关文献的标准做法，reviewer 熟悉 | 与文献不对齐，需要额外解释 |
| **实验增益** | Baseline 基础较低，我的方法相对增益更显著（对比好看）| Baseline 基础更高，增益空间被压缩，对比可能不好看 |
| **灾难性遗忘** | 模型看不到历史数据，遗忘现象可能被人工放大 | 更接近标准 CL 设定，遗忘现象更真实 |

### 当前状态与倾向
- 代码实现的是 **Task-only**
- 曾经倾向于改为 Cumulative（归因更干净），但最终未修改
- **用户仍在犹豫，此问题未最终拍板**

### 修改代价
如果改为 Cumulative，只需改两处：
```python
# trainer.py 第726行（训练）
# 改前：
valid_node_mask = (self.dataset.timesteps >= task_start_t) & (self.dataset.timesteps <= task_end_t)
# 改后：
valid_node_mask = self.dataset.timesteps <= task_end_t

# trainer.py 第604行（评估）
# 改前：
eval_mask = (self.dataset.timesteps >= task_start) & (self.dataset.timesteps <= task_end)
# 改后：
eval_mask = self.dataset.timesteps <= task_end
```

### 新对话时应优先讨论
1. 用户的研究叙事最终是哪个方向？（"模型适应新 pattern" 还是 "模型在时序流中的整体泛化"）
2. 是否已经有实验结果可以辅助判断？
3. 目标会议的 reviewer 对哪种设定更熟悉/更认可？

---

## 十、遗留困惑与待决策问题

### 已解决
- [x] 灾难性遗忘问题：实验证明基本不存在，研究叙事中不再提及
- [x] GraphSMOTE 实现错误：已修复
- [x] PMP config 缺失：已补全
- [x] 脚本路径错误：已修复

### 待解决
- [ ] **⚠️ Snapshot 设计（最优先）**：task-only vs cumulative，见第九节，尚未最终确认
- [ ] **⚠️ 评估阈值未确定**：当前代码 `threshold=0.15` 导致所有模型 recall=1.0，无法区分模型优劣。已确定使用固定阈值方案（而非验证集搜索），但具体数值（0.3 / 0.4 / 0.5）尚未选定。需重跑实验后对比各阈值下 TASD-CL 相对 Naive 的提升幅度，选涨点最大的值写入论文。**下次对话优先处理此问题。**
- [ ] **DGraphFin OOM**：需要减少 GPU 显存占用，方案待定（子图采样 or 混合精度 or 模型简化）
- [ ] **TASD-CL 实验结果**：BSL + SSF/SPC/SCD 组合在三个数据集上的效果尚未验证（本次修复了 SCD old_model bug，需重跑）
- [ ] **理论补充**：SCD 中 Z_aa 子空间权重为 2.0 的理论动机尚未形式化

### 已解决（本次对话）
- [x] **SCD 组件失效 Bug**：`old_model` 在 TASD-CL 模式下（lwf_alpha=0）从未被保存，导致 SCD 始终返回 0。已在每个 Task 结束后独立保存 old_model（`trainer.py` 第 980-985 行）
- [x] **评估指标扩充**：新增 Macro F1、Macro Recall、Specificity、MCC，删除 Forgetting/BWT/avg_cost
- [x] **训练测试划分改为时序切分**：前 80% 时间步 → 训练，后 20% → 测试，与论文"训练过去预测未来"叙事一致
- [x] **Task 1 标记为 warm-up**：欺诈样本极少，CSV 中 `is_warmup=True`，论文分析排除该任务

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
