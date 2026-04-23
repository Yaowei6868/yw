# fraud-detection-gnn 工作备忘（codex.md）

最后更新：2026-04-18

## 1. 当前论文的核心定位

这篇论文要解决的核心问题不是泛化地说“灾难性遗忘”，而是：

**在工业 task-only snapshot 构图约束下，历史图结构不可用，而 fraud pattern 会随时间演化；我们要在不依赖历史全图结构的前提下，持续保持稳定的图欺诈检测能力。**

更准确的叙事是：

- 工业场景下不能维护 cumulative global graph
- 因此每个 task 只能基于当前时间窗口构建 task-only snapshot
- 新 task 的训练会贴合当前 fraud pattern
- 历史 graph structure 在后续 task 中不可用
- 传统 graph replay / 通用 CL 在该设定下存在结构性局限

因此论文主线应该是：

**task-only constraint 下持续建模 evolving fraud patterns**

而不是：

**单纯解决 catastrophic forgetting**

## 2. 方法定位

### 2.1 TASD-CL 的准确定位

TASD-CL 是一个**语义感知的持续学习框架**，主实例化在 CGNN-style backbone 上。

不能写成：

- 我们提出了一个全新的 backbone
- 我们完全复现了原论文 CGNN 并直接在其上增量改造

更准确的说法是：

- 我们采用/改编了一个 **CGNN-style semantic decomposition backbone**
- 在该 backbone 上提出了 TASD-CL
- 核心贡献在 **SSF / SPC / SCD 三个组件**
- 这三个组件并不完全绑死在 CGNN 上，理论上也可迁移到 BSL 等具有可分解语义表征的 backbone
- 但 CGNN 提供的 `x_nor / x_abnor / alpha` 让该框架最自然、最完整

推荐英文表述方向：

- `CGNN-based backbone`
- `CGNN-style semantic decomposition backbone`
- `instantiated on a CGNN-style backbone`

避免写：

- `faithful reproduction of the original CGNN`
- `our newly proposed backbone`

### 2.2 三个组件的作用层面

- **A / SSF**：参数层。保护关键语义参数，尤其是 `lin_nor / lin_abnor / att_*`
- **B / SPC**：表征层。对 `x_nor / x_abnor` 提取高斯原型并做 structure-free replay
- **C / SCD**：语义层。利用旧模型的 `alpha` 过滤高置信 normal/abnormal 节点，再对子空间语义蒸馏

一句话总结：

- A 保参数
- B 保表征
- C 保语义

### 2.3 Framework 的正式定义

当前应将 TASD-CL 统一定义为：

**TASD-CL is a semantic-aware continual fraud detection framework under task-only snapshot constraints, instantiated on a CGNN-style backbone, and preserving knowledge at the parameter, representation, and semantic levels via SSF, SPC, and SCD.**

中文可统一表述为：

**TASD-CL 是一个面向 task-only snapshot 持续欺诈检测的语义感知框架，实例化在 CGNN-style backbone 上，并通过 SSF、SPC、SCD 在参数、表示、语义三个层面保持历史知识。**

该 framework 应拆成四个正式模块：

1. **Problem Setting**
   - streaming tasks
   - task-only snapshots
   - historical graph unavailable
   - evolving fraud patterns

2. **Backbone Instantiation**
   - CGNN-style semantic decomposition backbone
   - 暴露 `out / x_nor / x_abnor / alpha`

3. **Continual Knowledge Preservation**
   - SSF：参数层保持
   - SPC：表示层原型回放
   - SCD：语义层条件蒸馏

4. **Task-End Knowledge Update**
   - 更新 Fisher / 参数快照
   - 更新 prototype buffer
   - 更新 teacher model

### 2.4 总损失的统一写法

代码里实际 loss 项较多，但论文里必须统一收口，避免看起来像“堆 loss”。

推荐在论文中写成：

\[
\mathcal{L}_{total}
=
\mathcal{L}_{backbone}
+
\lambda_{ssf}\mathcal{L}_{ssf}
+
\lambda_{spc}\mathcal{L}_{spc}
+
\lambda_{scd}\mathcal{L}_{scd}
\]

其中：

- `L_backbone`
  - 当前任务监督损失 `L_task`
  - backbone 自身约束（如 CGNN 的 `loss_csd + loss_consis`）
- `L_ssf`
  - 参数层知识保持
- `L_spc`
  - 子空间原型回放
- `L_scd`
  - 条件子空间蒸馏

重要：**主文中不要把 engineering heuristics 和 framework core 混在一起写。**

应放到实现细节 / appendix 的内容包括：

- dynamic focal alpha
- warm-up task skip
- abnormal 分支额外权重
- `scd_tau`
- `spc_n_samples`
- 其他采样和阈值细节

主文只保留核心结构：

- `Backbone`
- `SSF`
- `SPC`
- `SCD`

## 3. CGNN 在当前项目中的定位

### 3.1 现在是否算用了 CGNN 作为 backbone

算。

原因：

- 当前实现保留了 CGNN 最核心的主干结构：
  - `lin_in`
  - `CGNNLayer`
  - `lin_nor`
  - `lin_abnor`
  - `att_lin`
  - `att_vec`
  - `classifier`

但必须承认：

- 当前实现不是原论文 CGNN 的 1:1 faithful reproduction
- 我们为了 task-only continual setting 和 TASD-CL 增加了缓存、replay 接口以及训练逻辑

因此最准确的结论是：

**我们使用的是 adapted CGNN backbone / CGNN-style backbone，而不是原论文静态 low-label setting 的完全复刻版 CGNN。**

### 3.2 对比实验是否站得住

站得住，但 claim 要写准。

成立的前提是：

- 所有方法都在**统一的 task-only continual setting**下评估
- 使用统一的 snapshot construction、task schedule、训练预算、指标
- 比较的是“谁更适应这个新设定”

因此论文里应该写：

- `We adapt representative fraud-detection baselines to a unified task-only continual evaluation protocol.`

不要写：

- `We exactly reproduce all original baselines.`

## 4. 当前论文中的实验逻辑

## 4.1 主实验逻辑

当前实验应该分为三层：

### 第一层：统一设定下的主结果对比

在统一的 task-only continual setting 下比较：

- CGNN
- BSL
- ConsisGAD
- GradGNN
- HOGRL
- PMP
- GCN（可作传统基线）
- TASD-CL（CGNN backbone）

目的不是复现原论文 static setting，而是验证：

**在统一 task-only continual fraud detection 设定下，TASD-CL 是否优于代表性图欺诈检测方法和通用 CL 方案。**

### 第二层：通用 CL 对比

必须补齐：

- Elliptic 上的 CGNN + EWC
- Elliptic 上的 CGNN + LwF
- Elliptic 上的 CGNN + ER

这是当前最优先事项，因为它直接回答：

**TASD-CL 是否优于通用持续学习策略，而不只是优于 naive。**

### 第三层：组件消融

要证明：

- A 有用
- B 有用
- C 有用

因此需要：

- `backbone + A + B + C`（full）
- `backbone + B + C`（noA / noSSF）
- `backbone + A + C`（noB / noSPC）
- `backbone + A + B`（noC / noSCD）

即：

- full
- noSSF
- noSPC
- noSCD

这正是当前 CGNN 消融的基本思路。

## 4.2 当前 CGNN 消融的解释方式

CGNN 消融不是“原论文 CGNN 的架构消融”，而是：

**在 CGNN backbone 上，对 TASD-CL 三个组件进行消融。**

对应配置：

- `configs/ours/main/elliptic_TASDCL_CGNN.yaml`
- `configs/ours/ablation/elliptic_TASDCL_noSSF_CGNN.yaml`
- `configs/ours/ablation/elliptic_TASDCL_noSPC_CGNN.yaml`
- `configs/ours/ablation/elliptic_TASDCL_noSCD_CGNN.yaml`

解释方式：

- noSSF = 关掉 A，仅保留 B + C
- noSPC = 关掉 B，仅保留 A + C
- noSCD = 关掉 C，仅保留 A + B

## 5. 当前已经理清楚的关键结论

### 5.1 方法图该怎么画

方法图的中心不应是“抗遗忘”，而应是：

**task-only constraint 下持续建模 evolving fraud patterns**

推荐主线：

1. 左侧：problem setting
   - streaming tasks
   - task-only snapshots
   - evolving fraud patterns
   - historical structure unavailable

2. 中间：CGNN semantic decomposition
   - `x_nor`
   - `x_abnor`
   - `alpha`

3. 周围：TASD-CL 三组件
   - 上：SSF
   - 下：SPC
   - 右：SCD

4. 最右：objective / outcome
   - stable continual fraud detection under task-only constraints

### 5.2 对审稿人的关键说法

论文中要强调：

- 我们比较的是**统一 task-only continual setting 下的适配结果**
- 不是原论文静态设定结果的逐字复现
- 我们的方法是 framework contribution，不是 backbone invention

### 5.3 当前最强的论文叙事

推荐统一表述：

**We study continual fraud detection under task-only snapshot construction, where historical graph structure is unavailable and fraud patterns evolve over time. To address this, we propose TASD-CL, a semantic-aware continual learning framework instantiated on a CGNN-style backbone. TASD-CL preserves fraud knowledge at the parameter, representation, and semantic levels through SSF, SPC, and SCD, respectively.**

### 5.4 当前最适合 CIKM 的方法卖点

面向 CIKM 2026，方法卖点应集中成三句，而不是展开成过长的实现细节：

- `We formulate continual fraud detection under task-only snapshot constraints.`
- `We preserve fraud knowledge at the parameter, representation, and semantic levels.`
- `We avoid historical graph replay by using structure-free subspace prototypes.`

更具体地说，论文主文中应优先突出：

- 新问题设定：task-only continual fraud detection
- 三层知识保持：SSF / SPC / SCD
- 不依赖历史图结构：structure-free replay

而不是优先突出：

- 复杂训练技巧
- 多个阈值与 warm-up 规则
- 代码层面的实现分支

## 6. 当前存在的主要风险点

### 风险 1：baseline 表述不严谨

不能把当前 `CGNN-Naive` 说成原论文 CGNN 的 faithful reproduction。

必须改成：

- adapted baseline
- unified continual setting baseline

### 风险 2：当前 CGNN 组件消融不是严格单变量控制

这是目前最需要修的实验问题。

当前观察到：

- full 配置使用 `spc_lambda=0.1`, `spc_n_samples=128`
- 但 `noSSF` 和 `noSCD` 仍然使用旧配置 `spc_lambda=0.3`, `spc_n_samples=32`

这意味着：

- 当前 Full vs noSSF / noSCD 的对比不只是“去掉一个组件”
- 还混入了 SPC 超参数变化

因此如果投稿，必须重跑一版**严格消融**：

- full：A+B+C
- noSSF：B+C，其他超参与 full 完全一致
- noSPC：A+C，其他超参与 full 完全一致
- noSCD：A+B，其他超参与 full 完全一致

### 风险 3：框架泛化性证据还不够强

虽然理论上 A/B/C 可迁移到 BSL 等 backbone，但如果要增强 framework claim，最好补一点证据：

- 至少展示在 BSL 上的一组简化迁移结果，或
- 在正文中明确：CGNN 是主实例化 backbone，BSL 结果仅说明一定迁移性

### 风险 4：当前 loss 表述不够收口，容易给人“堆模块 / 堆损失”的印象

目前代码中的总损失由：

- 当前任务监督
- backbone 自身约束
- SSF / EWC
- SPC replay
- SCD distillation
- 若干 engineering trick

共同组成。

这在实现层面没有问题，但在论文呈现上必须收口为：

- `L_backbone`
- `L_ssf`
- `L_spc`
- `L_scd`

否则 reviewer 很容易认为方法不够简洁，或者是“多个 loss 的经验性拼接”。

### 风险 5：需要补充效率 / 存储分析来强化 task-only setting 下的方法合理性

对于 CIKM，这一点值得单独补：

- SSF 额外存储：Fisher + old params
- SPC 额外存储：每 task / class / subspace 的 `(mu, sigma)`
- SCD 额外存储：一个 teacher snapshot

必须强调：

- **不存历史图结构**
- **不回放历史子图**
- **额外存储远小于 graph replay**

## 7. 当前最高优先事项

### 最高优先：补齐 Elliptic 上的通用 CL 基线

需要运行：

- `configs/ours/cl_on_cgnn/elliptic_EWC_CGNN.yaml`
- `configs/ours/cl_on_cgnn/elliptic_LwF_CGNN.yaml`
- `configs/ours/cl_on_cgnn/elliptic_ER_CGNN.yaml`

对应脚本：

- `scripts/run_elliptic_cl_baselines.sh`

原因：

- 这是当前主结论中最缺的一块证据链
- 没有它，论文只能说明“比 naive 强”，还不能充分说明“比通用 CL 更合适”

## 8. 投稿前必须补齐的实验清单（按优先级）

1. **运行 Elliptic CGNN 通用 CL 基线**
   - EWC / LwF / ER

2. **重跑严格 CGNN 消融**
   - 除去被移除的组件外，其余超参与 full 完全一致

3. **检查所有 baseline 在 unified task-only setting 下的公平性**
   - epochs
   - lr / tuning effort
   - task schedule
   - metrics

4. **视时间决定是否补 backbone 泛化证据**
   - 如 BSL 上的一小组验证

5. **统一论文叙事**
   - 从“防遗忘”改为“task-only constraint 下持续建模 evolving fraud patterns”

6. **把方法写法从“多项损失”收口成 framework**
   - 主文统一写成 `L_backbone + SSF + SPC + SCD`
   - engineering heuristics 放 implementation / appendix

7. **补一个简短但明确的效率 / 存储分析**
   - 对比 historical graph replay
   - 强调 structure-free prototype replay 的优势

8. **如版面允许，补一小段方法图图注 / pseudo algorithm**
   - current-task training
   - task-end state update
   - next-task knowledge reuse

## 9. 当前代码讲解进度（已完成主线讲解）

当前已经从代码层面讲清楚的部分如下，后续新对话应从这里继续，而不是重复从头开始。

### 9.1 已讲清楚的主线

#### A. 程序入口

已讲清：

- 所有实验都从 `train.py` 启动
- `train.py` 只做：
  - 读取 `--config`
  - `OmegaConf.load`
  - 初始化 `Trainer(c)`
  - 调用 `trainer.train()`
  - 最后 `trainer.save(c.name)`

结论：

- `train.py` 本身不决定实验方法
- 真正的实验逻辑由 yaml 和 `Trainer` 决定

#### B. `Trainer.__init__` 如何加载数据

已讲清：

- `elliptic` 走 `EllipticDataset(...).pyg_dataset()`
- `elliptic_actor` 走 `EllipticPlusActorDataset(...)[0]`
- `dgraphfin` 走 `DGraphFinDataset(...)[0]`
- 最终都转成一个 PyG `Data` 对象，保存在 `self.dataset`
- `self.dataset.x` 会统一转成 `float32`

结论：

- `self.dataset` 是整个实验的全局底库图
- 后面的 task 训练和评估都从这个全局底库图裁切

#### C. `self.dataset` 里到底装了什么

已讲清：

- `x`
- `edge_index`
- `y`
- `timesteps`
- `classified_idx`
- `unclassified_idx`

结论：

- `self.dataset` 不是某个 task 的图
- 它是全局图底库
- 所有 task 都基于它切出当前窗口节点和当前窗口边

#### D. `task_train_idx` / `task_valid_idx` 是怎么来的

已讲清 `_get_task_indices(time_steps)` 的完整逻辑：

1. 根据 `time_steps` 找出当前窗口里的所有节点
2. 只保留有标签节点
3. 按时间顺序做 80/20 切分：
   - 前 80% 时间 -> `task_train_idx`
   - 后 20% 时间 -> `task_valid_idx`
4. 若切分失败则退化为随机 80/20

结论：

- `task_train_idx` / `task_valid_idx` 是**全局节点索引**
- 它们的作用是决定：
  - 哪些节点参与监督训练
  - 哪些节点参与当前 task 验证

#### E. “切监督节点”和“切 snapshot 图结构”是两步不同操作

这部分已特别讲清：

- **切监督节点**
  - 由 `_get_task_indices()` 完成
  - 决定主损失和验证指标作用在哪些节点上

- **切 snapshot 图结构**
  - 在 `train()` 里通过 `valid_node_mask` 和 `edge_mask` 完成
  - 决定消息传播允许沿哪些边进行

结论：

- 当前 task 图中出现的节点，不一定都参与监督训练
- 当前 task 参与监督训练的节点，也不是靠“重新裁节点矩阵”得到的

#### F. `snapshot_data` 是怎么构造的

已讲清：

- `snapshot_data = copy.copy(self.dataset)`
- 只替换：
  - `snapshot_data.edge_index = self.dataset.edge_index[:, edge_mask]`
- 然后 `snapshot_data.to(device)`

结论：

- `snapshot_data` 不是完全重新编号的小图
- 它是：
  - **全局节点特征不变**
  - **当前窗口图结构生效**

也就是：

- 特征空间：全量
- 图结构：task-only
- 监督目标：当前 task 的 labeled train nodes

#### G. 一个 epoch 的最基础主训练逻辑

当前已经讲清 `cgnn` 训练中这四行代码的具体含义：

```python
outputs, x_nor, x_abnor = self.model(snapshot_data, return_decomposed=True)
outputs = outputs.reshape((self.dataset.x.shape[0]))
task_y = self.dataset.y[current_train_idx.cpu()].float().to(self.device).reshape(-1, 1)
task_loss = self.criterion(outputs[current_train_idx].reshape(-1, 1), task_y)
```

已经讲清：

- 模型对整个 `snapshot_data` 做 full-graph forward
- 得到：
  - `outputs`
  - `x_nor`
  - `x_abnor`
- `outputs` 先 reshape 成全局长度一维向量
- 再用 `current_train_idx` 从全局预测中取出当前训练节点的预测
- 再从全局标签中取出当前训练节点的真实标签
- 最后只在这些训练节点上算 `BCEWithLogitsLoss`

结论：

- 当前训练底板是：
  - **full-graph forward**
  - **subset supervised loss**

### 9.2 当前用户已经明确理解的点

用户已明确表示“明白了”的点：

- `task_train_idx / task_valid_idx` 不是随机乱抽，而是按时间切分出来的
- `Elliptic / Actor / DGraphFin` 在现实语义上不同，但可以统一抽象为时序图异常节点检测任务
- `Actor` 数据集节点更接近 actor/address entity，可在宽泛意义上视为 user-like entity，但不应和 DGraphFin 的真实业务用户节点混为一谈

### 9.3 当前代码讲解已完成到什么程度

截至目前，代码讲解的主线已经完成到：

- `CGNNLayer` 中 `x_nor / x_abnor / alpha` 的形成逻辑
- `message -> aggregate -> update` 的语义传播逻辑
- `outputs / x_nor / x_abnor / node_alpha` 的返回与作用
- trainer 中 `task_loss + cgnn_loss + SSF + SPC + SCD` 的拼装逻辑
- 三个组件在代码中各自如何运行

因此后续不应再继续逐行细抠代码，而应转向：

1. framework 定义
2. 方法章节写法
3. 实验逻辑收紧
4. 消融与 baseline 证据链补齐

若后续再回到代码层面，应只针对：

- 某个具体 bug
- 某个具体配置
- 某个组件的实现改动

而不是重新从头做逐行讲解。

## 10. 面向 CIKM 2026 的当前判断

如果目标是投稿 CIKM 2026，当前方法本身**可以成立**，但投稿前还需要把“证据链”和“呈现方式”同时收紧。

当前判断如下：

- **问题设定是成立的**
  - task-only continual fraud detection 有明确工业动机
- **framework 结构是成立的**
  - SSF / SPC / SCD 分层逻辑清楚
- **主要短板不在于继续加方法**
  - 而在于：
    - 通用 CL 基线未补齐
    - 消融不够严格
    - loss 表述不够收口
    - 缺少效率 / 存储分析

一句话判断：

**现在最需要做的不是继续增加新模块，而是把 framework 定义、实验控制和投稿叙事同时收紧。**

## 11. 写作时的硬性约束

论文中应始终坚持以下边界：

- 我们的方法贡献在 framework，不在 backbone 发明
- CGNN 是 adapted backbone，不是 faithful reproduction claim
- baseline 比较基于 unified task-only continual setting
- 消融实验必须是严格 controlled ablation

## 12. 三个数据集的统一理解

这三个数据集虽然都被纳入“图异常节点检测”框架，但它们的现实语义并不完全相同。

### 12.1 Elliptic

- **节点**：一笔比特币交易（transaction）
- **边**：交易之间的资金流转/输入输出关联关系
- **检测目标**：识别异常交易节点

这是一张**交易图**，异常对象是交易本身。

### 12.2 Elliptic++ Actor

- **节点**：地址 / actor 实体
- **边**：地址之间的交互、转账或关联关系
- **检测目标**：识别异常 actor / 地址节点

这是一张**实体图/地址图**，异常对象不再是单笔交易，而是参与交易的实体。

### 12.3 DGraphFin

- **节点**：用户
- **边**：用户之间的关系边（如联系人、社交、业务或其他用户关系）
- **检测目标**：识别异常/风险用户节点

这是一张**用户关系图**，异常对象是用户。

### 12.4 三者的共同抽象

虽然三个数据集的节点和边在现实意义上不同，但它们都可以统一抽象为：

- 节点具有特征 `x`
- 节点之间存在关系边 `edge_index`
- 节点具有正常/异常标签 `y`
- 节点带有时间信息 `timesteps`
- 目标是进行**异常节点检测**

因此，这三类数据都可以被统一表述成：

**temporal graph-based anomalous node detection**

也就是说，方法层面统一的是：

- 图结构学习任务形式相同
- 目标都是识别异常节点
- 都可以放进 task-only continual setting

而不是说：

- 节点现实语义完全相同
- 边现实语义完全相同

### 12.5 为什么可以用同一种方法

可以用同一种方法的原因是：

- GNN 方法依赖的是图上的共同数学结构，而不是节点现实名称
- 只要都能写成 `(V, E, X, Y)` 的节点分类问题，就可以采用统一的异常节点检测框架
- TASD-CL 依赖的是：
  - 节点特征
  - 图关系
  - 正常/异常语义可分
  - 时间任务序列

这些条件在三类数据集上都成立。

### 12.6 需要在论文中强调的严谨表述

不能写成：

- 三个数据集语义相同，因此同一方法自然适用

应该写成：

- 三个数据集虽然在节点和边的现实语义上不同（transaction / actor / user），但都可以统一建模为时序图上的异常节点检测问题，因此可以在统一的 task-only continual evaluation protocol 下进行比较。

推荐英文表述方向：

`Although the node and edge semantics differ across datasets, they can all be formulated as temporal graph-based anomalous node detection problems, where each node is associated with features, relational edges, and an anomaly label.`

## 13. 一句话版本

**当前最重要的不是继续改方法，而是把 framework、实验逻辑和投稿叙事同时收紧：补齐 Elliptic 上的通用 CL 基线、重跑严格 CGNN 消融、把总损失收口成 `Backbone + SSF + SPC + SCD`、并在论文里明确 CGNN 是 adapted backbone、所有 baseline 都是在统一 task-only continual setting 下比较。**
