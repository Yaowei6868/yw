# Final Core Claim

Date: 2026-05-07

Status: Final claim confirmed by the author. All LaTeX sections should follow this claim consistently.

## 中文核心主张

区块链欺诈数据不是静态图数据，而是具有明显时间演化特征的金融交易图。交易关系、账户行为和风险传播模式都会随时间不断变化。与此同时，区块链欺诈检测数据通常具有显著的类别不平衡特征，欺诈节点或欺诈交易在整体样本中只占少数，使模型容易偏向正常类并忽略高风险异常模式。欺诈信号不仅可能体现在单笔交易层面，也可能体现在地址或行为主体的长期交互结构中。因此，区块链欺诈检测不应被简单视为一次性的静态图分类问题，而应被建模为时间快照下的持续图欺诈检测问题。

已有工作为这一设定提供了重要动机。BRIGHT 指出，金融交易欺诈检测必须遵守时间因果性：交易和实体天然形成图结构，但模型在预测当前风险时不能访问未来信息，否则会产生时间信息泄漏。HMGNN 进一步指出，动态图中同时存在长期稳定的结构演化和短期局部的瞬时变化。这说明，在时间演化的欺诈图中，模型既需要适应新的局部欺诈模式，也需要保持对稳定欺诈语义的识别能力。

在真实工业风控场景中，模型通常无法长期维护包含全部历史节点、历史边和邻居结构的全量动态图。一方面，持续累积历史交易图会带来显著的存储、显存和计算开销；另一方面，在线欺诈检测对时效性要求较高，频繁查询和重建历史多跳邻居会引入较高推理延迟。此外，交易图建模还必须避免使用未来交易信息进行当前风险预测，否则会造成时间信息泄漏。

基于这些现实约束，本文研究 task-only snapshot setting：每个时间任务只允许访问当前时间窗口内的图快照，进入后续任务后，不保存、不访问、不重放历史节点、历史边、邻接矩阵或历史子图。

在 task-only snapshot 约束下，区块链欺诈检测面临的核心问题是语义概念漂移。随着时间推进，正常交易行为和异常欺诈行为的语义边界会不断变化；模型如果只根据当前图快照进行更新，容易过度贴合当前窗口中的局部交易模式，从而削弱对历史 fraud-discriminative semantics 的稳定识别能力。动态图中同时存在长期稳定的结构依赖和短期突发的局部变化。对于区块链欺诈检测而言，长期稳定部分对应相对持续的风险传播规律和交互模式，短期变化部分对应不断更新的欺诈策略和异常交易形态。因此，问题关键在于：如何在 task-only snapshot 约束下适应新的欺诈语义，同时保持对历史正常语义和异常语义的判别能力。

为此，本文提出 TASD-CL，一个面向 task-only temporal blockchain fraud detection 的语义感知持续图欺诈检测框架。论文对外统一使用 TASD-CL 作为方法名，不反复区分“TASD 是 backbone、TASD-CL 是完整 framework”。Temporal Adaptive Semantic Decomposition 是 TASD-CL 的核心语义分解机制：它将节点表示显式分解为 normal semantic subspace 和 abnormal semantic subspace，并通过 alpha-guided routing 进行风险感知的信息聚合和欺诈预测。通过这种语义分解，模型能够显式区分正常行为语义和异常欺诈语义，为后续的持续语义保持提供清晰接口。

基于这一语义分解接口，TASD-CL 进一步引入三个轻量级持续学习组件来应对时间演化中的语义概念漂移。第一，SSF 在参数层面对语义关键参数施加更强约束，使语义分解、子空间投影和路由相关参数在跨任务更新时保持稳定。第二，SPC 在表示层面保存 normal/abnormal 子空间中的轻量级高斯原型，通过 structure-free semantic prototype consolidation 保留历史语义分布，而不保存任何历史图结构。第三，SCD 在语义层面利用旧模型的路由置信度，选择性蒸馏高置信正常语义和异常语义，从而增强模型对演化欺诈语义的连续建模能力。三者共同作用，使模型能够在不回放历史图结构的条件下持续适应新的欺诈模式。

实验上，本文以 AUC-ROC 和 MacroF1 作为主指标。由于区块链欺诈检测天然存在类别不平衡，单纯依赖整体准确率难以反映模型对少数欺诈类的识别能力。AUC-ROC 衡量模型对风险节点的整体排序能力，MacroF1 衡量类别不平衡条件下正常类和欺诈类的均衡检测能力。Elliptic 和 Elliptic++ Actor 分别对应 transaction-level 和 actor/address-level 的区块链欺诈图。虽然两个数据集中的节点与边语义不同，但它们都可以统一表述为时间演化区块链图上的节点风险识别任务。Elliptic 从交易粒度验证模型对交易级异常模式的识别能力，Elliptic++ Actor 从行为主体粒度验证模型对地址或账户层面长期交互风险的建模能力。二者在图粒度上互补，能够共同支撑本文关于 task-only temporal blockchain fraud detection 的统一实验口径。

## 统一写作口径

- 不把本文问题表述为 catastrophic forgetting 问题。
- 不在论文中报告 forgetting rate、backward transfer 或类似遗忘指标。
- 持续学习组件的主要动机是应对 task-only snapshot setting 下的 semantic concept drift，并保持 fraud-discriminative semantics。
- 区块链欺诈检测数据具有显著类别不平衡，论文中需要把这一点作为选择 MacroF1 的重要理由。
- 主实验指标使用 AUC-ROC 和 MacroF1。
- 论文对外统一使用 TASD-CL 作为方法名。
- Temporal Adaptive Semantic Decomposition 是 TASD-CL 的核心语义分解机制，不在论文中反复表述为一个单独的 backbone 方法。
- 不反复强调“TASD 是 backbone、TASD-CL 是完整 framework”，避免审稿人误解为两个独立方法或质疑二者边界。
- 不把 TASD-CL 写成 CGNN 的变体，也不写成 CGNN + TASD-CL。
- GCN 是 traditional graph neural network，代表传统图学习方法。
- CGNN、HOGRL、GradGNN、BSL、PMP 是 fraud detection baselines。
- GCN + EWC/LwF/ER 是 ordinary GCN 上的通用 continual learning baselines。
- Elliptic 和 Elliptic++ Actor 是互补的两个时间演化区块链欺诈图数据集，分别验证 transaction-level 和 actor/address-level 风险识别能力。

## TASD-CL 命名

论文对外统一使用 **TASD-CL** 作为方法名。TASD-CL 中的 TASD 指 **Temporal Adaptive Semantic Decomposition**，即时间自适应语义分解；CL 指 continual graph fraud detection setting 下的持续语义保持。

由于方法名中包含 **CL**，审稿人可能会自然联想到通用 continual learning 中的 catastrophic forgetting、forgetting rate、backward transfer 和 forward transfer 等传统评估口径。为了避免贡献边界被误解，论文中不应将 TASD-CL 定义为一个泛化的 continual learning 方法，而应明确限定为面向 task-only temporal blockchain fraud detection 的 continual graph fraud detection 方法。

这个命名保留了 **CL** 对 continual setting 的指向，但将问题范围明确限定在 continual graph fraud detection，而不是泛化到所有 continual learning benchmarks。论文中应强调 TASD-CL 解决的是 task-only snapshot 约束下的 semantic concept drift 和 fraud-discriminative semantics preservation，而不是把本文包装成解决通用 catastrophic forgetting 的方法。

因此，后续论文中统一使用:
- Method: **TASD-CL**
- Expanded description: **a Temporal Adaptive Semantic Decomposition framework for continual graph fraud detection**
- Mechanism: **temporal adaptive semantic decomposition**, including normal/abnormal semantic subspaces and alpha-guided routing
