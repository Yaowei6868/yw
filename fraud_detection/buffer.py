"""
维护一个“记忆”仓库，用来存放过去任务中的一小部分代表性节点
它被设计为“按类别平衡”的，以确保少数类（如 Illicit 节点）不会在记忆中被多数类（Licit 节点）淹没
"""
import numpy as np
import torch

class ReplayBuffer:
    """
    一个按类别平衡的回放缓冲区。
    它为每个类别存储固定数量的样本（节点索引）。
    """
    def __init__(self, buffer_size_per_class: int):
        """
        初始化缓冲区。
        buffer_size_per_class: 每个类别（Licit/Illicit）要存储的最大样本数。
        例如，如果 buffer_size_per_class 被设置为 100，
        那么缓冲区将最多存储 100 个 Licit 节点（标签为 0）和 100 个 Illicit 节点（标签为 1）。总缓冲区大小将是 200。
        """
        self.buffer_size_per_class = buffer_size_per_class
        # 缓冲区按类别存储节点索引 (0: Licit, 1: Illicit)
        #' 映射 '1' 为 1, '2' 为 0]
        self.buffer = {0: [], 1: []}
        self.seen_classes = {0, 1} # 我们关心的数据集中的类别只有0和1，对于2不考虑

    def add_exemplars(self, node_indices: np.ndarray, node_labels: np.ndarray):
        """
        这是更新缓冲区的核心方法。它在一个任务训练完成之后被调用，从该任务的训练集中采样并添加 exemplars 到缓冲区。
        node_indices: 刚训练完的任务的训练节点索引
        node_labels: 对应的节点标签
        """

        """
        1. 将新节点按类别分开
        初始化一个字典，代码遍历这些新节点，并将它们的索引按照标签（0或1）放入 new_nodes_by_class 字典中。
        例如：{0: [100, 102, 105, ...], 1: [101, 108, ...]}。
        """
        new_nodes_by_class = {c: [] for c in self.seen_classes}
        for idx, label in zip(node_indices, node_labels):
            label = int(label) # 确保是整数
            if label in self.seen_classes:
                new_nodes_by_class[label].append(idx)

        """
        2. 为每个类别更新缓冲区
        """
        for label, new_nodes in new_nodes_by_class.items():
            if not new_nodes: # 如果这个类别没有新节点，则跳过
                continue

            # 获取当前类别的“记忆仓库”（例如，self.buffer[0]，即 Licit 节点的存储列表）。    
            current_class_buffer = self.buffer[label]
            
            # 关键步骤。将新任务的节点随机打乱。这确保了我们接下来添加或替换的样本是随机的，而不是按某种（可能有偏见的）顺序。
            np.random.shuffle(new_nodes)

            """
            3. 计算需要填充多少
            nodes_to_add：我们最多只想添加 buffer_size_per_class 个新节点。
            如果新任务中该类的节点少于缓冲区大小，我们就只添加所有可用的新节点。
            """
            nodes_to_add = min(len(new_nodes), self.buffer_size_per_class)
            free_space = self.buffer_size_per_class - len(current_class_buffer) # 缓冲区还可以添加的空间
            
            """
            4. 填充缓冲区
            如果缓冲区还有空闲空间（free_space > 0），我们就先填充这些空间。
            添加的节点数量（add_now）是新节点中最多可以添加的数量（nodes_to_add）和空闲空间中最多可以添加的数量（free_space）中的较小值。
            例如，如果新任务中该类的节点少于缓冲区大小，我们就只添加所有可用的新节点。
            """
            if free_space > 0:
                add_now = min(nodes_to_add, free_space)
                current_class_buffer.extend(new_nodes[:add_now])
                # 移除已添加的节点
                new_nodes = new_nodes[add_now:]
                nodes_to_add -= add_now
            
            """
            5. 如果缓冲区已满，且还有新节点要添加，则执行随机替换
            这确保了缓冲区始终包含来自所有过去任务的混合样本，并且较新的任务有机会将其样本存入内存，而不会完全覆盖掉最早的记忆。
            如果缓冲区已满（free_space <= 0），我们就需要从缓冲区中随机选择一些节点进行替换。
            我们要替换的节点数量（nodes_to_add）是新节点中最多可以添加的数量（nodes_to_add）和缓冲区中最多可以替换的数量（free_space）中的较小值。
            例如，如果新任务中该类的节点少于缓冲区大小，我们就只替换所有可用的节点。
            """
            if nodes_to_add > 0:
                # 随机选择缓冲区中的位置进行替换
                replace_indices = np.random.choice(
                    range(len(current_class_buffer)), 
                    nodes_to_add, 
                    replace=False
                )
                
                # 执行替换
                for i, node_idx in zip(replace_indices, new_nodes):
                    current_class_buffer[i] = node_idx

    def get_buffer_indices(self) -> torch.Tensor:
        """
        获取缓冲区中存储的所有节点索引，用于合并到当前任务的训练集中。
        这是一个简单的获取器（Getter）方法。它在下一个新任务开始训练时被调用。
        :return: 一个包含所有回放节点索引的 PyTorch 张量。
        """
        all_indices = self.buffer[0] + self.buffer[1]
        
        if not all_indices: # 如果缓冲区为空（例如在第一个任务中）
            return torch.tensor([], dtype=torch.long)
            
        return torch.tensor(all_indices, dtype=torch.long)


class SubspacePrototypeBuffer:
    """
    TASD-CL Component B: Subspace Prototype Condensation (SPC) Buffer.

    在 BSL 的三个解耦子空间 (Z_na, Z_aa, Z_nn) 中分别存储 Gaussian 原型，
    而非原始节点索引。这解决了图回放的根本缺陷：历史节点的图上下文（邻居）
    已随时间步变化，直接回放原始节点会引入结构噪声。
    原型仅依赖子空间表示，与图结构无关。

    存储代价: O(T * 2 * 3 * sub_dim)  vs  O(K * feat_dim) 原始回放
    """

    def __init__(self, sub_dim: int):
        """
        sub_dim: BSL 每个子空间的维度 = hidden_dim // 3
        """
        self.sub_dim = sub_dim
        # prototypes[task_id][class_id][subspace_k] = (mu: Tensor[sub_dim], sigma: Tensor[sub_dim])
        self.prototypes = {}

    def add_task_prototypes(self, task_id: int, z_parts: list, y_labels):
        """
        在任务 t 结束后，计算并存储各子空间的类原型。

        z_parts: list of 3 tensors, each [n_train, sub_dim], BSL get_sub_features() 的输出
        y_labels: [n_train] 训练节点标签 (0/1, 不含 -1)
        """
        self.prototypes[task_id] = {}
        for cls in [0, 1]:
            cls_mask = (y_labels == cls)
            if cls_mask.sum() < 2:
                continue
            self.prototypes[task_id][cls] = {}
            for k in range(3):
                z_cls = z_parts[k][cls_mask].detach().cpu()  # [n_cls, sub_dim]
                mu = z_cls.mean(dim=0)                        # [sub_dim]
                sigma = z_cls.std(dim=0).clamp(min=1e-6)      # [sub_dim]
                self.prototypes[task_id][cls][k] = (mu, sigma)

    def sample_prototypes(self, n_per_class_per_task: int = 32, device: str = 'cpu'):
        """
        从所有已存储任务的原型中采样，生成回放用的合成 z 向量。

        返回:
            z_replay: [N_replay, hidden_dim]  (3 个子空间拼接)
            y_replay: [N_replay]  (0/1 标签)
        若缓冲区为空则返回 (None, None)。
        """
        if not self.prototypes:
            return None, None

        z_list, y_list = [], []
        for task_id, task_protos in self.prototypes.items():
            for cls, cls_protos in task_protos.items():
                if len(cls_protos) < 3:
                    continue
                subspace_samples = []
                for k in range(3):
                    mu, sigma = cls_protos[k]
                    mu = mu.to(device)
                    sigma = sigma.to(device)
                    eps = torch.randn(n_per_class_per_task, self.sub_dim, device=device)
                    subspace_samples.append(mu.unsqueeze(0) + eps * sigma.unsqueeze(0))
                # 拼接三个子空间 -> [n, hidden_dim]
                z_list.append(torch.cat(subspace_samples, dim=1))
                y_list.append(torch.full(
                    (n_per_class_per_task,), cls, dtype=torch.long, device=device
                ))

        if not z_list:
            return None, None
        return torch.cat(z_list, dim=0), torch.cat(y_list, dim=0)

    def has_prototypes(self) -> bool:
        return len(self.prototypes) > 0