import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
import copy 
import time 
import pandas as pd 
import gc 
from torch_geometric.utils import to_dense_adj, subgraph

# 引入所有模型
from .models import (
    GAT, GCN, GIN, GraphSAGE, STAGNN, EvolveGCN, TGN, MLP, 
    GATv2, HOGRL, CGNN, GradGNN, BSL, PMPModel, ConsisGAD
)
from .datasets import EllipticDataset, EllipticPlusActorDataset
from .buffer import ReplayBuffer 

# 在 trainer.py 顶部添加此类
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        :param alpha: 控制正负样本权重的平衡因子 (0 < alpha < 1)。
                      如果 alpha=0.75，则正样本权重为 0.75，负样本为 0.25。
        :param gamma: 聚焦参数，gamma 越大，模型越关注难分样本。
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # BCEWithLogitsLoss 包含了 Sigmoid 层，数值更稳定
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 计算 pt: 如果 y=1, pt=p; 如果 y=0, pt=1-p
        pt = torch.exp(-bce_loss) 
        
        # 构建 alpha 因子 tensor
        # targets 形状为 [N, 1] 或 [N]
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = 1.0
            
        # Focal Loss 公式: -alpha * (1-pt)^gamma * log(pt)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 模型映射表
models_map = {
    "gcn": GCN,
    "gat": GAT,
    "gatv2": GATv2,
    "gin": GIN,
    "graphsage": GraphSAGE,
    "stagnn": STAGNN,
    "evolvegcn": EvolveGCN,
    "tgn": TGN,
    "mlp": MLP,
    "hogrl": HOGRL,       
    "cgnn": CGNN,         
    "grad": GradGNN,      
    "bsl": BSL,           
    "pmp": PMPModel,      
    "consisgad": ConsisGAD, 
    "gat_cobo": GATv2,    
    "fraudgnn_rl": GATv2  
}

datasets_map = {
    "elliptic": EllipticDataset,
    "elliptic_actor": EllipticPlusActorDataset
}


class Trainer:
    # 初始化训练器
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
        # 实例化数据集 (适配 Elliptic++ Actor)
        if self.config.train.dataset == 'elliptic_actor':
            self.dataset_obj = EllipticPlusActorDataset(root='data/elliptic++actor')
            # 先放在 CPU 上，等到 train loop 里再把 snapshot 搬运到 GPU
            self.dataset = self.dataset_obj[0]
        else:
            # 兼容旧逻辑
            self.dataset_obj = datasets_map[self.config.train.dataset](config.dataset)
            self.dataset = self.dataset_obj.pyg_dataset().to(self.device)
            
        self.config.model.input_dim = self.dataset.num_node_features

        # 初始化模型
        self.model = models_map[self.config.train.model](config.model).to(self.device)
        
        # 计算 Loss 权重
        all_labels = self.dataset.y
        valid_mask = all_labels != -1
        y_valid = all_labels[valid_mask]

        default_weight = 3.0 
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([default_weight]).to(self.device)
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
        self.tensorboard = SummaryWriter(log_dir=os.path.join('runs', self.config.name)) 

        # [CL 机制初始化]
        self.replay_buffer = ReplayBuffer(config.train.get('buffer_size_per_class', 0))
        self.ewc_lambda = config.train.get('ewc_lambda', 0.0) 
        self.ewc_params = {}     
        self.ewc_fisher = {}     
        self.lwf_alpha = config.train.get('lwf_alpha', 0.0)
        self.lwf_temperature = config.train.get('lwf_temperature', 1.0)
        self.old_model = None    

        # [CL 评估专用]
        self.num_tasks = len(self.config.train.task_schedule)
        self.f1_matrix = np.zeros((self.num_tasks, self.num_tasks))
        
        # [任务划分]
        self.task_schedule = config.train.task_schedule
        self.task_valid_indices_map = {} 
        self.recall_matrix = [] 
        self.aggregate_metrics_history = []
        
        
        # [初始化任务划分]
        if hasattr(self.dataset, 'timesteps'):
            self.timesteps = self.dataset.timesteps
        else:
            # 兜底逻辑
            self.timesteps = (torch.arange(self.dataset.num_nodes) * 10 // self.dataset.num_nodes).to(self.device)
        
        self.task_indices = {}
        for t in range(10): 
            mask_t = (self.timesteps == t)
            valid_mask = (self.dataset.y != -1)
            self.task_indices[t] = torch.where(mask_t & valid_mask)[0]

    # 计算评估指标
    def compute_metrics(self, preds, labels, threshold=0.3):
        pred_labels = (preds > threshold).astype(int)
        
        # 1. 混淆矩阵基础 (用于计算 Cost 和 G-Mean)
        try:
            tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
        except ValueError:
            tn, fp, fn, tp = 0, 0, 0, 0

        # --- [指标 1, 2, 3] Binary Metrics (只关注欺诈类 Class 1) ---
        precision = precision_score(labels, pred_labels, pos_label=1, zero_division=0)
        recall = recall_score(labels, pred_labels, pos_label=1, zero_division=0) # Binary Recall
        f1 = f1_score(labels, pred_labels, pos_label=1, zero_division=0)  # Binary F1

        # --- [指标 4, 8] Ranking Metrics (排序能力) ---
        try:
            auc_roc = roc_auc_score(labels, preds)
            auc_pr = average_precision_score(labels, preds)
        except ValueError:
            auc_roc = 0.0
            auc_pr = 0.0

        # --- [指标 5, 6] Macro Metrics (宏平均，关注整体公平性) ---
        # Macro Recall = (Recall_Class0 + Recall_Class1) / 2
        macro_recall = recall_score(labels, pred_labels, average='macro', zero_division=0)
        # Macro F1 = (F1_Class0 + F1_Class1) / 2
        macro_f1 = f1_score(labels, pred_labels, average='macro', zero_division=0)

        # --- [指标 7] G-Mean (几何平均，衡量正负类平衡) ---
        # Specificity (Class 0 的 Recall) = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        # G-Mean = sqrt(Sensitivity * Specificity)
        g_mean = np.sqrt(recall * specificity)

        # --- [指标 9] Cost (财务代价) ---
        cost_fn = 100.0  # 漏抓代价
        cost_fp = 1.0    # 误抓代价
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        avg_cost = total_cost / len(labels) if len(labels) > 0 else 0.0
        
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "g_mean": g_mean,
            "total_cost": total_cost,
            "avg_cost": avg_cost
        }

    # 知识蒸馏损失
    def _distillation_loss(self, student_output, teacher_output):
        prob_s = student_output.squeeze().clamp(min=1e-7, max=1-1e-7)
        prob_t = teacher_output.detach().squeeze().clamp(min=1e-7, max=1-1e-7) 
        kl_div = prob_t * (torch.log(prob_t) - torch.log(prob_s)) + \
                 (1 - prob_t) * (torch.log(1 - prob_t) - torch.log(1 - prob_s))
        return kl_div.mean()

    # CGNN 专用损失函数
    def _cgnn_loss(self, x_nor, x_abnor, y, train_idx):
        """
        [CGNN] 计算辅助损失
        1. CSD Loss: 强制 Normal 部分和 Abnormal 部分正交（互不相关）
        2. Consistency Loss: 特征能量约束
        """
        # 只在训练节点上计算
        x_nor_train = x_nor[train_idx]
        x_abnor_train = x_abnor[train_idx]
        
        # --- 1. CSD Loss (Discrepancy) ---
        # 目标：最小化 x_nor 和 x_abnor 的余弦相似度 -> 也就是让它们正交
        # Normalize
        nor_norm = F.normalize(x_nor_train, p=2, dim=1)
        abnor_norm = F.normalize(x_abnor_train, p=2, dim=1)
        # Cosine Similarity
        cos_sim = (nor_norm * abnor_norm).sum(dim=1)
        # Loss = mean(cos_sim^2)
        loss_csd = (cos_sim ** 2).mean()
        
        # --- 2. Consistency / Feature Constraint ---
        # 这一步是为了防止分解退化。我们希望：
        # 对于 Normal 节点 (y=0): x_nor 的能量应大于 x_abnor
        # 对于 Fraud 节点 (y=1): x_abnor 的能量应大于 x_nor
        y_train = y[train_idx]
        
        # 计算 L2 Norm
        norm_n = torch.norm(x_nor_train, p=2, dim=1)
        norm_a = torch.norm(x_abnor_train, p=2, dim=1)
        
        # Margin Ranking Loss logic
        margin = 0.5
        # y=0: max(0, norm_a - norm_n + margin) -> 希望 n > a
        loss_0 = F.relu(norm_a - norm_n + margin)
        # y=1: max(0, norm_n - norm_a + margin) -> 希望 a > n
        loss_1 = F.relu(norm_n - norm_a + margin)
        
        loss_consis = torch.mean(torch.where(y_train == 0, loss_0, loss_1))
        
        return loss_csd, loss_consis

    # 对比学习损失
    def _sup_contrastive_loss(self, features, labels, temperature=0.07):
        labels = labels.view(-1)
        sim_matrix = torch.matmul(features, features.T) / temperature
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach() 
        batch_size = labels.shape[0]
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(features.device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(features.device), 0)
        mask = mask * logits_mask
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-7)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)
        return -mean_log_prob_pos.mean()

    # EWC: 计算 Fisher 信息矩阵
    def _update_ewc_metrics(self, task_train_idx: torch.Tensor, dataset):
        """
        计算 EWC 的 Fisher 信息矩阵 (修复设备不匹配问题)
        """
        if self.ewc_lambda <= 0.0: return     
        print("   [EWC] Updating Fisher Matrix...")
        self.model.eval() 
        self.optimizer.zero_grad()      
        # 1. 前向传播 (在 GPU 上)
        # HOGRL 兼容处理
        if self.config.train.model == 'hogrl' and not hasattr(dataset, 'adjs'):
             order = self.config.model.get('num_orders', 3)
             dataset.adjs = self._precompute_high_order_graphs(
                dataset.edge_index, self.dataset.num_nodes, order=order
             )
        
        # PMP/ConsisGAD 特殊处理: EWC update 需要完整 forward
        # 注意: GradGNN 这种有多分支的需要小心，这里暂时只走默认 forward
        out_res = self.model(dataset)     
        
        if isinstance(out_res, tuple): 
            outputs = out_res[0] 
        else: 
            outputs = out_res           
        outputs = outputs.reshape((-1,))      
        # 2. 准备标签 (关键修复)
        # self.dataset.y 通常在 CPU，而 task_train_idx 在 GPU
        # 必须先将索引转回 CPU 才能去取 CPU 上的标签
        idx_cpu = task_train_idx.cpu()
        task_y = self.dataset.y[idx_cpu].float().to(self.device).reshape(-1, 1)    
        # 3. 准备预测值
        # outputs 已经在 GPU 上，task_train_idx 也在 GPU 上，直接索引即可
        pred = outputs[task_train_idx].reshape(-1, 1)
        # 4. 计算梯度并累积 Fisher
        loss = self.criterion(pred, task_y)
        loss.backward() 
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if name not in self.ewc_fisher: 
                    self.ewc_fisher[name] = torch.zeros_like(param.data).to(param.device)
                
                # Fisher = gradients^2
                self.ewc_fisher[name].data += param.grad.data.pow(2)
                # 备份参数用于后续正则化计算
                self.ewc_params[name] = param.data.clone()
    
    # EWC: 获取任务节点索引
    def _get_task_indices(self, time_steps: list):
        start_time = time_steps[0]
        end_time = time_steps[-1]
        
        task_mask = (self.dataset.timesteps >= start_time) & (self.dataset.timesteps <= end_time)
        task_nodes_idx = np.where(task_mask.cpu().numpy())[0]
        classified_nodes_global_idx = self.dataset.classified_idx 
        if torch.is_tensor(classified_nodes_global_idx):
            classified_nodes_global_idx = classified_nodes_global_idx.cpu().numpy()
        task_classified_mask = np.isin(task_nodes_idx, classified_nodes_global_idx)
        task_classified_idx = task_nodes_idx[task_classified_mask] 
        if len(task_classified_idx) == 0: return None, None 
        task_train_idx, task_valid_idx = train_test_split(task_classified_idx, test_size=0.15, random_state=42) 
        return torch.tensor(task_train_idx, dtype=torch.long).to(self.device), \
               torch.tensor(task_valid_idx, dtype=torch.long).to(self.device)

    # HOGRL 专用: 预计算高阶邻接矩阵 (COO 格式)
    def _precompute_high_order_graphs(self, edge_index, num_nodes, order=3):
        print(f"Pre-computing HOGRL graphs up to order {order} (Force COO)...")
        device = self.device
        from torch_geometric.utils import add_self_loops
        edge_index_self, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_self
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(edge_index_self, values, (num_nodes, num_nodes)).coalesce()
        adjs = [] 
        current_power = adj
        a_powers = [current_power]
        for k in range(2, order + 1):
            next_power = torch.sparse.mm(current_power, adj)
            next_power = next_power.coalesce()
            a_powers.append(next_power)
            current_power = next_power
            print(f"  - A^{k} computed (COO).")
        return a_powers
    
    # GradGNN: 节点分组采样
    def _node_group_sampling(self, num_nodes, group_size=32):
        """
        [Grad] Algorithm 3: Node Group Sampling
        """
        indices = torch.randperm(num_nodes)
        num_groups = num_nodes // group_size
        groups = []
        for i in range(num_groups):
            groups.append(indices[i*group_size : (i+1)*group_size])
        return groups, indices # indices 用于后续映射回原图
    
    def _grad_gcl_loss(self, z, y, temperature=0.2):
        """
        [Grad] Eq. 2: Supervised Contrastive Loss
        """
        # 简化版实现，避免过大的内存消耗
        # 随机采样一部分节点计算 loss
        batch_size = 1024
        if z.size(0) > batch_size:
            idx = torch.randperm(z.size(0))[:batch_size]
            z = z[idx]
            y = y[idx]
            
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / temperature
        exp_sim = torch.exp(sim)
        
        # Mask for same class
        mask = torch.eq(y.view(-1, 1), y.view(-1, 1)).float().to(z.device)
        # Remove diagonal
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0)).to(z.device)
        
        mask = mask * logits_mask
        exp_sim = exp_sim * logits_mask
        
        # Sum of exp(sim) for all negatives/positives
        denominator = exp_sim.sum(dim=1, keepdim=True)
        
        # Log Probability
        log_prob = sim - torch.log(denominator + 1e-8)
        
        # Mean over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        return -mean_log_prob_pos.mean()
    
    # BSL: 基于节点分组的对比学习   辅助函数：根据节点标签确定边类型
    def _get_edge_types(self, edge_index, y):
        """
        返回 labeled 节点之间的边及其类型。
        Types: 0: NA (Normal-Abnormal), 1: AA (Abnormal-Abnormal), 2: NN (Normal-Normal)
        """
        src, dst = edge_index
        y_src = y[src]
        y_dst = y[dst]
        
        # 只保留两端都有标签的边 (Labeled Edges)
        mask_labeled = (y_src != -1) & (y_dst != -1)
        src_l, dst_l = src[mask_labeled], dst[mask_labeled]
        y_s, y_d = y_src[mask_labeled], y_dst[mask_labeled]
        
        # 判定类型
        # NA: y_s != y_d
        # AA: y_s == 1 and y_d == 1
        # NN: y_s == 0 and y_d == 0
        edge_types = torch.zeros_like(y_s).long() # 默认为 0 (NA)
        
        # AA (1+1=2)
        mask_aa = (y_s == 1) & (y_d == 1)
        edge_types[mask_aa] = 1
        
        # NN (0+0=0) -> 设为 2 (为了对应 embedding 的索引顺序: NA=0, AA=1, NN=2)
        mask_nn = (y_s == 0) & (y_d == 0)
        edge_types[mask_nn] = 2
        
        # NA (1+0 or 0+1) -> 设为 0
        mask_na = (y_s != y_d)
        edge_types[mask_na] = 0
        
        return src_l, dst_l, edge_types
    
    # 核心损失计算逻辑
    def _compute_bsl_full_loss(self, model, data, outputs, z_all, alpha_all, train_idx, valid_node_mask):
        """
        完全复现论文 Eq. 23: L = L_class + alpha * L_D + beta * L_bsl
        其中 L_D = L_link + L_attn
             L_bsl = L_con + L_incon
        """
        device = self.device
        y = data.y
        
        # --- 1. Edge Classification Loss (L_link) ---
        # 仅使用有标签节点的边
        src_l, dst_l, edge_types = self._get_edge_types(data.edge_index, y)
        
        if len(src_l) > 0:
            # 随机采样一部分边以防计算量过大 (可选)
            if len(src_l) > 2048:
                idx = torch.randperm(len(src_l))[:2048]
                src_l, dst_l, edge_types = src_l[idx], dst_l[idx], edge_types[idx]

            # 预测边得分。这里简化实现：我们只计算 正确类型子空间 的得分
            z_src = z_all[src_l]
            z_dst = z_all[dst_l]
            
            # 获取所有 3 个子空间的 logits
            z_src_parts = model.get_sub_features(z_src)
            z_dst_parts = model.get_sub_features(z_dst)
            
            link_losses = []
            for k in range(3): # 0:NA, 1:AA, 2:NN
                feat_cat = torch.cat([z_src_parts[k], z_dst_parts[k]], dim=1)
                logits = model.edge_decoder(feat_cat).squeeze()
                
                # Label: 如果当前 k 等于真实类型 edge_types，则为 1，否则为 0
                target = (edge_types == k).float()
                link_losses.append(F.binary_cross_entropy_with_logits(logits, target))
            
            l_link = sum(link_losses)
        else:
            l_link = torch.tensor(0.0).to(device)

        # --- 2. Weight Contrastive Loss (L_attn) ---
        # Eq. 9: 正常节点(y=0) NN权重应>AA; 异常节点(y=1) AA权重应>NN
        alpha_train = alpha_all[train_idx]
        y_train = y[train_idx].float()
        
        # alpha columns: 0:NA, 1:AA, 2:NN
        margin = 0.2
        loss_norm = F.relu(alpha_train[:, 1] - alpha_train[:, 2] + margin) # AA - NN
        loss_fraud = F.relu(alpha_train[:, 2] - alpha_train[:, 1] + margin) # NN - AA
        
        l_attn = torch.mean((1 - y_train) * loss_norm + y_train * loss_fraud)
        
        l_d = l_link + l_attn

        # --- 3. Barely Supervised Learning (L_bsl) ---
        # 需要采样 Anchor (Labeled) 和 Unlabeled 节点
        
        # A. 采样 Anchor (Labeled Normal n, Labeled Fraud a)
        pos_mask = (y[train_idx] == 1)
        neg_mask = (y[train_idx] == 0)
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return l_d, torch.tensor(0.0).to(device) # 无法进行 BSL
            
        # 随机选一个 n 和 一个 a
        idx_n = train_idx[neg_mask][torch.randint(0, neg_mask.sum(), (1,))]
        idx_a = train_idx[pos_mask][torch.randint(0, pos_mask.sum(), (1,))]
        
        z_n = z_all[idx_n] # [1, D]
        z_a = z_all[idx_a] # [1, D]
        
        # B. 采样 Unlabeled
        # 在当前 Task 的 Valid Node 中，排除掉 train_idx 即为 Unlabeled
        all_indices = torch.where(valid_node_mask)[0]
        # 简单去重：假设 train_idx 很小，随机采样的概率重叠很低，或者使用 mask 过滤
        unlabeled_indices = all_indices[torch.randperm(len(all_indices))[:256]] # Batch size for consistency
        
        z_u = z_all[unlabeled_indices] # [B, D]
        
        # C. 构造增强 (Augmentation)
        # Helper to reconstruct z from parts
        def reconstruct(z_parts_list):
            # 1. 重新计算 alpha
            new_alpha = model.get_attention_weights(z_parts_list)
            # 2. 加权拼接
            z_w = torch.cat([
                z_parts_list[0] * new_alpha[:, 0:1],
                z_parts_list[1] * new_alpha[:, 1:2],
                z_parts_list[2] * new_alpha[:, 2:3]
            ], dim=1)
            # 3. 预测
            return model.classifier(z_w)

        # 分解 Unlabeled 和 Anchors
        zu_parts = model.get_sub_features(z_u) # (na, aa, nn)
        zn_parts = model.get_sub_features(z_n)
        za_parts = model.get_sub_features(z_a)
        
        # 广播 Anchor
        zn_parts_exp = [p.expand(z_u.size(0), -1) for p in zn_parts]
        za_parts_exp = [p.expand(z_u.size(0), -1) for p in za_parts]

        # Weak Augmentation (Consistency): Replace NA (Index 0)
        # Z_nw = [Zn_na, Zu_aa, Zu_nn]
        z_nw_parts = [zn_parts_exp[0], zu_parts[1], zu_parts[2]]
        # Z_aw = [Za_na, Zu_aa, Zu_nn]
        z_aw_parts = [za_parts_exp[0], zu_parts[1], zu_parts[2]]
        
        pred_nw = reconstruct(z_nw_parts)
        pred_aw = reconstruct(z_aw_parts)
        
        # L_con = MSE(pred_nw, pred_aw)
        l_con = F.mse_loss(torch.sigmoid(pred_nw), torch.sigmoid(pred_aw))
        
        # Strong Augmentation (Inconsistency): Replace AA/NN (Index 1, 2)
        # Z_ns = [Zu_na, Zn_aa, Zn_nn]
        z_ns_parts = [zu_parts[0], zn_parts_exp[1], zn_parts_exp[2]]
        # Z_as = [Zu_na, Za_aa, Za_nn]
        z_as_parts = [zu_parts[0], za_parts_exp[1], za_parts_exp[2]]
        
        pred_ns = reconstruct(z_ns_parts)
        pred_as = reconstruct(z_as_parts)
        
        # L_incon = -MSE(pred_ns, pred_as)
        l_incon = -F.mse_loss(torch.sigmoid(pred_ns), torch.sigmoid(pred_as))
        
        l_bsl = l_con + 0.5 * l_incon # 权重可调
        
        return l_d, l_bsl

    # ConsisGAD: 预训练生成器和判别器
    def _consisgad_loss_consistency(self, p_orig, p_aug):
        """
        L_c (Eq. 3 & Eq. 9 part 1): KL Divergence or MSE between predictions
        """
        prob_orig = torch.sigmoid(p_orig)
        prob_aug = torch.sigmoid(p_aug)
        
        # 使得增强后的预测尽可能接近原始预测 (的高置信度伪标签)
        loss = F.mse_loss(prob_aug, prob_orig.detach())
        return loss

    def _consisgad_loss_diversity(self, h_orig, h_aug):
        """
        L_d (Eq. 9 part 2): Distance between representations.
        We want to MAXIMIZE diversity -> Minimize -Distance
        """
        # 使用欧氏距离
        dist = torch.norm(h_orig.detach() - h_aug, p=2, dim=1).mean()
        return -dist # 最小化负距离 = 最大化距离
    
    def _train_lga_step(self, snapshot_data):
        """[ConsisGAD] Pre-training"""
        opt_G = optim.Adam(self.model.generator.parameters(), lr=0.01)
        opt_D = optim.Adam(self.model.discriminator.parameters(), lr=0.01)
        x, edge_index = snapshot_data.x, snapshot_data.edge_index
        
        opt_D.zero_grad()
        score_real = self.model.discriminator(x, edge_index)
        prob_adj = self.model.generator(x, edge_index)
        adj_fake = self.model.generator.sample_adj(prob_adj)
        score_fake = self.model.discriminator(x, edge_index, edge_weight=adj_fake.detach())
        loss_d = -torch.mean(torch.log(score_real + 1e-8) + torch.log(1 - score_fake + 1e-8))
        loss_d.backward(); opt_D.step()
        
        opt_G.zero_grad()
        score_fake_g = self.model.discriminator(x, edge_index, edge_weight=adj_fake)
        l_adv = -torch.mean(torch.log(score_fake_g + 1e-8))
        l_reg = -F.mse_loss(prob_adj, torch.ones_like(prob_adj))
        loss_g = l_adv + 0.1 * l_reg
        loss_g.backward(); opt_G.step()
        return loss_d.item(), loss_g.item()

    # -------------------------------------------------------------
    # 核心评估逻辑：计算遗忘度和平均性能
    # -------------------------------------------------------------
    def evaluate_cl_metrics(self, current_task_id, task_duration):
        print(f"\n--- [CL Evaluation] Evaluating on all seen tasks (0 to {current_task_id}) ---")
        self.model.eval()
        # 1. 初始化指标列表
        row_metrics = {
            "f1": [], "recall": [], "precision": [],
            "auc_roc": [], "auc_pr": [], "g_mean": [], 
            "avg_cost": [] 
        }

        # === 核心循环：逐个回顾历史任务 ===
        for t_id in range(current_task_id + 1):
            
            # A. 检查验证集是否存在
            if t_id not in self.task_valid_indices_map:
                for k in row_metrics: row_metrics[k].append(0.0)
                continue

            # B. 构建 Snapshot (防 OOM)
            task_start = self.task_schedule[t_id][0]
            task_end = self.task_schedule[t_id][-1]
            
            # 严格限制在当前测试任务的时间窗口内
            eval_mask = (self.dataset.timesteps >= task_start) & (self.dataset.timesteps <= task_end)
            
            # 构建子图：只保留两端都在窗口内的边
            row, col = self.dataset.edge_index
            edge_mask = eval_mask[row] & eval_mask[col]
            
            # 复制并搬运到 GPU
            snapshot_data = copy.copy(self.dataset)
            snapshot_data.edge_index = self.dataset.edge_index[:, edge_mask]
            snapshot_data = snapshot_data.to(self.device) 

            # C. 推理 (No Grad)
            with torch.no_grad():
                # HOGRL 特殊处理
                if self.config.train.model == 'hogrl':
                    order = self.config.model.get('num_orders', 3)
                    snapshot_data.adjs = self._precompute_high_order_graphs(
                        snapshot_data.edge_index, self.dataset.num_nodes, order=order
                    )
                
                # 模型前向传播
                # 兼容不同模型的返回格式
                if self.config.train.model == 'cgnn':
                    outputs, _, _ = self.model(snapshot_data, return_decomposed=True)
                elif self.config.train.model == 'bsl':
                    outputs, _, _ = self.model(snapshot_data, return_stats=True)
                elif self.config.train.model == 'consisgad':
                    outputs = self.model(snapshot_data, augment=False)
                elif self.config.train.model == 'grad':
                    outputs, _ = self.model(snapshot_data)
                else:
                    out_res = self.model(snapshot_data)
                    if isinstance(out_res, tuple): outputs = out_res[0] 
                    else: outputs = out_res
                
                outputs = outputs.reshape((self.dataset.x.shape[0]))
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                labels_all = self.dataset.y.numpy()

                # D. 只取该任务验证集的部分
                valid_idx = self.task_valid_indices_map[t_id]
                t_preds = probs[valid_idx]
                t_labels = labels_all[valid_idx]
                
                # E. 计算指标 (降低阈值以提高Recall，针对不平衡数据)
                res = self.compute_metrics(t_preds, t_labels, threshold=0.15)
                
                # F. 填入结果矩阵 (用于后续计算 Forgetting 和 BWT)
                if hasattr(self, 'f1_matrix'):
                    self.f1_matrix[current_task_id, t_id] = res['f1']
                
                # 记录以便算平均
                for k in row_metrics:
                    if k in res:
                        row_metrics[k].append(res[k])

            # G. 清理显存
            del snapshot_data, outputs
            torch.cuda.empty_cache()

        # === 2. 计算平均指标 (Average Performance) ===
        avg_metrics = {k: np.mean(v) for k, v in row_metrics.items()}
        
        # === 3. 计算 记忆能力指标 (Forgetting & BWT) ===
        avg_forgetting = 0.0
        avg_bwt = 0.0
        
        # 只有当完成了至少一个旧任务后 (Task 1 以后)，才能计算遗忘和迁移
        if current_task_id > 0 and hasattr(self, 'f1_matrix'):
            forgetting_sum = 0.0
            bwt_sum = 0.0
            
            # 遍历所有旧任务 (j < current_task_id)
            for j in range(current_task_id): 
                # --- Forgetting (越低越好) ---
                history_best = self.f1_matrix[:current_task_id, j].max()
                current_score = self.f1_matrix[current_task_id, j]
                forgetting_sum += (history_best - current_score)
                
                # --- Backward Transfer (BWT) (越高越好) ---
                original_score = self.f1_matrix[j, j] # 对角线分数
                bwt_sum += (current_score - original_score)
            
            avg_forgetting = forgetting_sum / current_task_id
            avg_bwt = bwt_sum / current_task_id

        # === 4. 打印日志 ===
        print(f"*** CL Metrics @ Task {current_task_id+1} ***")
        print(f"  [Accuracy]  Avg F1: {avg_metrics['f1']:.4f} | AUC-ROC: {avg_metrics['auc_roc']:.4f} | G-Mean: {avg_metrics['g_mean']:.4f}")
        print(f"  [Stability] Avg Forgetting (↓): {avg_forgetting:.4f} | Avg BWT (↑): {avg_bwt:.4f}")
        
        if hasattr(self, 'f1_matrix'):
            current_row = self.f1_matrix[current_task_id, :current_task_id+1]
            print(f"  > Matrix Row:  {np.round(current_row, 4)}")
        
        # Tensorboard 记录
        for k, v in avg_metrics.items():
            self.tensorboard.add_scalar(f"CL/Avg_{k}", v, current_task_id + 1)
        self.tensorboard.add_scalar("CL/Avg_Forgetting", avg_forgetting, current_task_id + 1)
        self.tensorboard.add_scalar("CL/Avg_BWT", avg_bwt, current_task_id + 1)

        # === 5. 返回结果 (存入 CSV) ===
        result_entry = {
            "task_id": current_task_id + 1,
            "time_cost": task_duration,
            "avg_forgetting": avg_forgetting,
            "avg_bwt": avg_bwt,
            **{f"avg_{k}": v for k, v in avg_metrics.items()}
        }
        return result_entry

    # -------------------------------------------------------------
    # 主训练循环
    # -------------------------------------------------------------
    def train(self):
        # 获取每个任务的 epoch 数 (默认为 100)
        epochs_per_task = self.config.train.get('num_epochs_per_task', 50) 
        global_step = 0
        start_time_total = time.time() 

        # --- 任务循环 (Task Loop) ---
        for task_id, time_steps in enumerate(self.task_schedule):
            print(f"\n--- Training on Task {task_id + 1} (Timesteps: {time_steps[0]} to {time_steps[-1]}) ---")
            task_start_time = time.time() 
            
            # 1. 获取当前任务的数据索引
            task_train_idx, task_valid_idx = self._get_task_indices(time_steps)
            if task_train_idx is None:
                print("Skipping task: No labeled data.")
                continue
            
            # 记录验证集索引用于评估
            self.task_valid_indices_map[task_id] = task_valid_idx.cpu().numpy()
            
            # 2. 动态计算 Loss 权重 (处理类别不平衡)
            y_curr = self.dataset.y[task_train_idx.cpu()]
            num_pos = (y_curr == 1).sum().item()
            num_neg = (y_curr == 0).sum().item()
            raw_ratio = num_neg / num_pos if num_pos > 0 else 1.0
            
            # === [修复顺序] 先计算权重，再打印 ===
            if task_id == 0:
                clip_cap = 20.0  # Task 1 冷启动，给予更高权重上限
            else:
                clip_cap = 10.0  # 后续任务恢复正常，防止过拟合
            
            # 计算变量
            clipped_weight = max(min(raw_ratio, clip_cap), 1.0) 
            
            # 现在可以安全打印了
            print(f"🔧 Task {task_id+1} 动态权重: {raw_ratio:.2f} -> 截断为: {clipped_weight:.2f}")
            
            dynamic_alpha = clipped_weight / (1.0 + clipped_weight)
            print(f"   -> Focal Loss Alpha: {dynamic_alpha:.4f}")
            
            # 更新 Loss 函数
            self.criterion = BinaryFocalLoss(alpha=dynamic_alpha, gamma=2.0).to(self.device)

            # 3. 构建 Snapshot (当前任务子图)
            # 作用: 仅保留当前时间窗口内的节点和边，防止显存爆炸 (OOM)
            task_start_t, task_end_t = time_steps[0], time_steps[-1]
            valid_node_mask = (self.dataset.timesteps >= task_start_t) & (self.dataset.timesteps <= task_end_t)
            row, col = self.dataset.edge_index
            edge_mask = valid_node_mask[row] & valid_node_mask[col]
            
            snapshot_data = copy.copy(self.dataset)
            snapshot_data.edge_index = self.dataset.edge_index[:, edge_mask]
            snapshot_data = snapshot_data.to(self.device) # 此时才搬运到 GPU
            print(f"📉 Snapshot 节点数: {valid_node_mask.sum().item()} (仅当前任务)")

            # 4. 特定模型的预计算步骤
            is_ewc_mode = self.ewc_lambda > 0.0
            is_lwf_mode = self.lwf_alpha > 0.0
            is_replay_mode = self.replay_buffer.buffer_size_per_class > 0

            # [HOGRL] 预计算高阶邻接矩阵 A^2, A^3...
            if self.config.train.model == 'hogrl':
                print(f">>> [HOGRL] 正在为 Task {task_id+1} 预计算高阶图...")
                order = self.config.model.get('num_orders', 3)
                snapshot_data.adjs = self._precompute_high_order_graphs(
                    snapshot_data.edge_index, self.dataset.num_nodes, order=order
                )
            
            # [ConsisGAD] 预训练 LGA 模块 (如果是旧版实现保留此逻辑，新版无需)
            if self.config.train.model == 'consisgad' and hasattr(self.model, 'discriminator'):
                print(">>> [ConsisGAD] 预训练 LGA 模块...")
                for lga_ep in range(20): 
                    ld, lg = self._train_lga_step(snapshot_data)

            # --- Epoch 循环 (训练主逻辑) ---
            for epoch in range(1, epochs_per_task + 1):
                global_step += 1
                self.model.train()
                self.optimizer.zero_grad()
                
                # 合并当前任务数据 + Replay Buffer 数据
                replay_idx = self.replay_buffer.get_buffer_indices().to(self.config.train.device)
                current_train_idx = torch.cat([task_train_idx, replay_idx])
                
                # ==========================================
                # 分支 A: Grad 模型特殊训练流程
                # ==========================================
                if self.config.train.model == 'grad':
                    # 1. [cite_start]训练 GCL 编码器 (监督对比学习) [cite: 741-766]
                    z_gcl = self.model.forward_gcl(snapshot_data)
                    gcl_labels = self.dataset.y[current_train_idx.cpu()].to(self.device)
                    loss_gcl = self._grad_gcl_loss(z_gcl[current_train_idx], gcl_labels)
                    
                    # 2. [cite_start]训练扩散模型 (关系生成) [cite: 767-807]
                    # 节点分组采样 (Node Group Sampling)
                    groups, perm_idx = self._node_group_sampling(snapshot_data.num_nodes, self.model.group_size)
                    
                    # [关键修复] 随机采样部分 Group 以防止 OOM
                    num_sample_groups = 64
                    if len(groups) > num_sample_groups:
                        sampled_grp_idx = torch.randperm(len(groups))[:num_sample_groups]
                        batch_groups = [groups[i] for i in sampled_grp_idx]
                    else:
                        batch_groups = groups

                    adj_batch_list = []
                    for grp_nodes in batch_groups:
                        grp_nodes = grp_nodes.to(self.device)
                        # 提取子图，显式传入 num_nodes 防止越界
                        sub_edge_index, _ = subgraph(
                            grp_nodes, 
                            snapshot_data.edge_index, 
                            relabel_nodes=True, 
                            num_nodes=snapshot_data.num_nodes
                        )
                        # 转为 Dense 矩阵
                        dense_adj = to_dense_adj(sub_edge_index, max_num_nodes=self.model.group_size)[0]
                        adj_batch_list.append(dense_adj)
                    
                    adj_batch = torch.stack(adj_batch_list)
                    
                    # 采样时间步 t
                    current_batch_size = adj_batch.size(0)
                    t = torch.randint(0, self.model.diff_steps, (current_batch_size,), device=self.device)
                    noise = torch.randn_like(adj_batch)
                    
                    # 扩散过程 (加噪)
                    alpha_cumprod = torch.linspace(0.99, 0.01, self.model.diff_steps).to(self.device)
                    alpha_t = alpha_cumprod[t].view(-1, 1, 1)
                    noisy_adj = torch.sqrt(alpha_t) * adj_batch + torch.sqrt(1 - alpha_t) * noise
                    
                    # 预测噪声
                    noise_pred = self.model.denoise_net(noisy_adj, t)
                    loss_diff = F.mse_loss(noise_pred, noise)
                    
                    # 3. [cite_start]训练检测器 (Beta Wavelet) [cite: 823-835]
                    out, _ = self.model(snapshot_data, generated_adj=None)
                    outputs = out.reshape((self.dataset.x.shape[0]))
                    
                    task_y = self.dataset.y[current_train_idx.cpu()].float().to(self.device).reshape(-1, 1)
                    loss_det = self.criterion(outputs[current_train_idx].reshape(-1, 1), task_y)
                    
                    # 总 Loss: 检测 + GCL + 扩散
                    total_loss = loss_det + 0.1 * loss_gcl + 0.1 * loss_diff
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step(total_loss)

                # ==========================================
                # 分支 B: ConsisGAD 模型特殊训练流程 (双层交替优化)
                # ==========================================
                elif self.config.train.model == 'consisgad':
                    # 初始化增强器的优化器
                    if not hasattr(self, 'aug_optimizer'):
                        self.aug_optimizer = optim.Adam(
                            self.model.augmentor.parameters(), 
                            lr=self.config.train.lr, 
                            weight_decay=self.config.train.weight_decay
                        )
                    
                    # 1. 前向传播 (同时获取原图和增强图的输出)
                    out_real, out_aug, h_real, h_aug = self.model(snapshot_data, augment=True)
                    
                    outputs = out_real.reshape((self.dataset.x.shape[0]))
                    out_aug = out_aug.reshape((self.dataset.x.shape[0]))
                    
                    # 2. [cite_start]筛选 "高质量无标签节点" (High Quality Nodes) [cite: 1312]
                    # === [新增] Warm-up 策略 ===
                    # 前 10 个 Epoch 处于“预热期”，只进行纯监督训练，不被伪标签误导
                    warmup_epochs = 10 
                    is_warmup = epoch <= warmup_epochs
                    
                    mask_hq = torch.zeros(outputs.shape[0], dtype=torch.bool, device=self.device)
                    
                    if not is_warmup:
                        probs = torch.sigmoid(outputs).detach() # detach 用于筛选掩码
                        all_indices = torch.arange(snapshot_data.num_nodes, device=self.device)
                        is_unlabeled = ~torch.isin(all_indices, current_train_idx)
                        
                        # 维持较低的阈值
                        tau_n, tau_a = 0.70, 0.60
                        mask_hq = ((probs < (1 - tau_n)) | (probs > tau_a)) & is_unlabeled
                    
                    # ==========================================
                    # Step A: 训练增强器 (Augmentor)
                    # ==========================================
                    if not is_warmup and mask_hq.sum() > 0:
                        l_consist = self._consisgad_loss_consistency(outputs[mask_hq].detach(), out_aug[mask_hq])
                        l_diver = self._consisgad_loss_diversity(h_real[mask_hq].detach(), h_aug[mask_hq])
                        loss_aug = l_consist + l_diver
                        
                        self.aug_optimizer.zero_grad()
                        loss_aug.backward() # 不需要 retain_graph，因为 Step B 会重算
                        self.aug_optimizer.step()
                    
                    # ==========================================
                    # Step B: 训练 GNN (Backbone)
                    # =========================================       
                    self.optimizer.zero_grad()
                    
                    # 重新计算 Backbone 的输出 (只计算 GNN 部分，不走 Augmentor)
                    # 这样可以构建一个新的计算图用于 GNN 更新
                    out_real_new = self.model(snapshot_data, augment=False)
                    outputs_new = out_real_new.reshape((self.dataset.x.shape[0]))
                    
                    # 1. Supervised Loss
                    task_y = self.dataset.y[current_train_idx.cpu()].float().to(self.device).reshape(-1, 1)
                    loss_sup = self.criterion(outputs_new[current_train_idx].reshape(-1, 1), task_y)
                    
                    # 2. Consistency Loss
                    loss_consist_gnn = 0.0
                    if not is_warmup and mask_hq.sum() > 0:
                        pseudo_labels = (probs[mask_hq] > 0.5).float().detach()
                        
                        # 关键：我们需要 GNN 对 "增强后的特征" 产生的预测
                        # 此时我们使用 Step A 产生的 h_aug (并 detach 掉，视为固定输入)
                        h_aug_fixed = h_aug.detach() 
                        
                        # 将这个固定的增强特征喂给 GNN 的分类器
                        # 这需要 models.py 支持直接输入 embedding，或者我们手动调用 classifier
                        out_aug_new = self.model.classifier(h_aug_fixed)
                        out_aug_new = out_aug_new.reshape((self.dataset.x.shape[0]))
                        
                        loss_consist_gnn = F.binary_cross_entropy_with_logits(
                            out_aug_new[mask_hq].view(-1, 1), 
                            pseudo_labels.unsqueeze(1)
                        )
                    
                    total_loss = loss_sup + 0.5 * loss_consist_gnn
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step(total_loss)

                # ==========================================
                # 分支 C: 标准模型训练流程 (BSL, PMP, GNNs)
                # ==========================================
                else:
                    # PMP 掩码注入
                    if self.config.train.model == 'pmp':
                        pmp_mask = torch.zeros(self.dataset.num_nodes, dtype=torch.bool, device=self.device)
                        pmp_mask[current_train_idx] = True
                        self.dataset.pmp_mask = pmp_mask 

                    # 前向传播
                    z_all = None; alpha_all = None
                    
                    # [修改] CGNN 特殊处理: 获取分解特征
                    if self.config.train.model == 'cgnn':
                        outputs, x_nor, x_abnor = self.model(snapshot_data, return_decomposed=True)
                    elif self.config.train.model == 'bsl':
                         outputs, z_all, alpha_all = self.model(snapshot_data, return_stats=True)
                    else:
                        out_res = self.model(snapshot_data)
                        if isinstance(out_res, tuple): outputs = out_res[0]
                        else: outputs = out_res
                    
                    outputs = outputs.reshape((self.dataset.x.shape[0]))
                    task_y = self.dataset.y[current_train_idx.cpu()].float().to(self.device).reshape(-1, 1)
                    
                    # 计算基础分类 Loss
                    task_loss = self.criterion(outputs[current_train_idx].reshape(-1, 1), task_y)
                    
                    bsl_loss, cl_loss, cgnn_loss = 0.0, 0.0, 0.0
                    
                    # [CGNN] 完整 Loss 计算
                    if self.config.train.model == 'cgnn':
                        # 获取配置中的权重 (默认 0.1)
                        w_csd = self.config.train.get('cgnn_lambda', 0.1)
                        w_consist = self.config.train.get('cgnn_beta', 0.1)
                        
                        l_csd, l_consist = self._cgnn_loss(x_nor, x_abnor, self.dataset.y.to(self.device), current_train_idx)
                        cgnn_loss = w_csd * l_csd + w_consist * l_consist

                    # [BSL] 完整 Loss 计算
                    if self.config.train.model == 'bsl' and z_all is not None:
                        l_d, l_bsl_term = self._compute_bsl_full_loss(
                            self.model, snapshot_data, outputs, z_all, alpha_all, 
                            current_train_idx, valid_node_mask
                        )
                        bsl_loss = 0.4 * l_d + 0.8 * l_bsl_term
                    
                    # [CL] 增量学习正则化 (LwF / EWC)
                    if is_lwf_mode and task_id > 0: 
                        self.old_model.eval()
                        with torch.no_grad():
                            old_out = self.old_model(snapshot_data) 
                            if isinstance(old_out, tuple): old_out = old_out[0]
                        cl_loss += self.lwf_alpha * self._distillation_loss(
                            torch.sigmoid(outputs[current_train_idx]), torch.sigmoid(old_out.reshape(-1)[current_train_idx]))
                    
                    if is_ewc_mode and task_id > 0: 
                        ewc_term = 0.0
                        for name, param in self.model.named_parameters():
                            if name in self.ewc_fisher:
                                ewc_term += (self.ewc_fisher[name].to(param.device) * (param - self.ewc_params[name].to(param.device)).pow(2)).sum()
                        cl_loss += self.ewc_lambda * ewc_term
                    
                    total_loss = task_loss + cl_loss + bsl_loss + cgnn_loss
                
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step(total_loss)

            # --- 任务结束后续处理 (Task End) ---
            # 1. EWC 更新 Fisher 矩阵
            if is_ewc_mode: self._update_ewc_metrics(task_train_idx, snapshot_data)
            # 2. LwF 更新旧模型
            if is_lwf_mode:
                self.old_model = copy.deepcopy(self.model)
                self.old_model.to(self.config.train.device)
            # 3. Replay Buffer 更新
            if is_replay_mode:
                idx_cpu = task_train_idx.cpu()
                task_train_labels = self.dataset.y[idx_cpu].numpy() 
                self.replay_buffer.add_exemplars(idx_cpu.numpy(), task_train_labels)

            task_duration = time.time() - task_start_time
            
            # 清理显存
            del task_train_idx, task_valid_idx, snapshot_data 
            gc.collect(); torch.cuda.empty_cache()

            # 评估当前任务及历史任务
            metrics_entry = self.evaluate_cl_metrics(task_id, task_duration)
            metrics_entry["cl_mode"] = "EWC" if is_ewc_mode else ("LwF" if is_lwf_mode else ("Replay" if is_replay_mode else "Naive"))
            self.aggregate_metrics_history.append(metrics_entry)
            gc.collect(); torch.cuda.empty_cache()

        print("All tasks trained.")
        total_time = time.time() - start_time_total
        print(f"Total time: {total_time:.2f}s")
        
        # 保存结果
        if self.aggregate_metrics_history:
            df = pd.DataFrame(self.aggregate_metrics_history)
            os.makedirs(os.path.join(self.config.train.save_dir, 'metrics'), exist_ok=True)
            df.to_csv(os.path.join(self.config.train.save_dir, f'metrics/{self.config.name}_aggregate_metrics.csv'), index=False)

        self.save(self.config.name)

    def test(self, dataset=None, labeled_only=False, threshold=0.5):
        dataset = dataset or self.dataset
        self.model.eval()
        out_res = self.model(dataset)
        if isinstance(out_res, tuple): out_res = out_res[0]
        outputs = out_res.reshape((dataset.x.shape[0]))
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = probs if labeled_only else probs[dataset.test_idx]
        return preds, preds > threshold

    def save(self, file_name):
        file_name = f"{file_name}.pt" if ".pt" not in file_name else file_name
        save_path = os.path.join(self.config.train.save_dir, file_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)