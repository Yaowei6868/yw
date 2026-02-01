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
import networkx as nx
import matplotlib.pyplot as plt
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
from .datasets import EllipticDataset,EllipticPlusActorDataset
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
    
    # BSL: 基于节点分组的对比学习
    def _get_bsl_augmentation(self, z_unlabeled, z_n, z_a, model):
        zu_na, zu_aa, zu_nn = model.get_sub_features(z_unlabeled) 
        zn_na, zn_aa, zn_nn = model.get_sub_features(z_n.unsqueeze(0))
        za_na, za_aa, za_nn = model.get_sub_features(z_a.unsqueeze(0))
        z_nw = torch.cat([zn_na.expand_as(zu_na), zu_aa, zu_nn], dim=1)
        z_aw = torch.cat([za_na.expand_as(zu_na), zu_aa, zu_nn], dim=1)
        z_ns = torch.cat([zu_na, zn_aa.expand_as(zu_aa), zn_nn.expand_as(zu_nn)], dim=1)
        z_as = torch.cat([zu_na, za_aa.expand_as(zu_aa), za_nn.expand_as(zu_nn)], dim=1)
        return z_nw, z_aw, z_ns, z_as

    # ConsisGAD: 预训练生成器和判别器
    def _train_lga_step(self):
        """[ConsisGAD] Pre-training"""
        opt_G = optim.Adam(self.model.generator.parameters(), lr=0.01)
        opt_D = optim.Adam(self.model.discriminator.parameters(), lr=0.01)
        x = self.dataset.x
        edge_index = self.dataset.edge_index
        
        opt_D.zero_grad()
        score_real = self.model.discriminator(x, edge_index)
        prob_adj = self.model.generator(x, edge_index)
        adj_fake = self.model.generator.sample_adj(prob_adj)
        score_fake = self.model.discriminator(x, edge_index, edge_weight=adj_fake.detach())
        loss_d = -torch.mean(torch.log(score_real + 1e-8) + torch.log(1 - score_fake + 1e-8))
        loss_d.backward(); opt_D.step()
        
        opt_G.zero_grad()
        prob_adj = self.model.generator(x, edge_index)
        adj_fake = self.model.generator.sample_adj(prob_adj)
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
        # 包含了你要的分类准确性指标: f1, auc_roc, g_mean
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
                
                # E. 计算指标 (使用 0.5 阈值以获得平衡的 F1/G-Mean)
                res = self.compute_metrics(t_preds, t_labels, threshold=0.6)
                
                # F. 填入结果矩阵 (用于后续计算 Forgetting 和 BWT)
                # self.f1_matrix 在 __init__ 中初始化: self.f1_matrix = np.zeros((10, 10))
                if hasattr(self, 'f1_matrix'):
                    self.f1_matrix[current_task_id, t_id] = res['f1']
                
                # 记录以便算平均
                for k in row_metrics:
                    if k in res:
                        row_metrics[k].append(res[k])

            # G. 清理显存
            del snapshot_data, out_res, outputs
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
                # 定义: max(历史上该任务最高分) - 当前分
                # 切片 [:current_task_id, j] 表示任务 j 在之前所有时刻的表现
                history_best = self.f1_matrix[:current_task_id, j].max()
                current_score = self.f1_matrix[current_task_id, j]
                forgetting_sum += (history_best - current_score)
                
                # --- Backward Transfer (BWT) (越高越好) ---
                # 定义: 当前分 - 刚学完该任务时的分 (R_{i,j} - R_{j,j})
                # BWT > 0 代表学习新任务反而让旧任务变好了 (正迁移)
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
        epochs_per_task = self.config.train.get('num_epochs_per_task', 100) 
        global_step = 0
        start_time_total = time.time() 

        for task_id, time_steps in enumerate(self.task_schedule):
            print(f"\n--- Training on Task {task_id + 1} (Timesteps: {time_steps[0]} to {time_steps[-1]}) ---")
            task_start_time = time.time() 
            task_train_idx, task_valid_idx = self._get_task_indices(time_steps)
            if task_train_idx is None:
                print("Skipping task: No labeled data.")
                continue
            self.task_valid_indices_map[task_id] = task_valid_idx.cpu().numpy()
            
            # --- Dynamic Weight Calculation ---
            y_curr = self.dataset.y[task_train_idx.cpu()]
            num_pos = (y_curr == 1).sum().item()
            num_neg = (y_curr == 0).sum().item()
            raw_ratio = num_neg / num_pos if num_pos > 0 else 1.0
            clipped_weight = max(min(raw_ratio, 10.0), 1.0)
            print(f"🔧 Task {task_id+1} Dynamic Weight: {raw_ratio:.2f} -> Clipped: {clipped_weight:.2f}")
            dynamic_alpha = clipped_weight / (1.0 + clipped_weight)
            print(f"   -> Focal Loss Alpha: {dynamic_alpha:.4f}")
            self.criterion = BinaryFocalLoss(alpha=dynamic_alpha, gamma=2.0).to(self.device)

            # --- Snapshot Construction ---
            task_start_t, task_end_t = time_steps[0], time_steps[-1]
            valid_node_mask = (self.dataset.timesteps >= task_start_t) & (self.dataset.timesteps <= task_end_t)
            row, col = self.dataset.edge_index
            edge_mask = valid_node_mask[row] & valid_node_mask[col]
            snapshot_data = copy.copy(self.dataset)
            snapshot_data.edge_index = self.dataset.edge_index[:, edge_mask]
            snapshot_data = snapshot_data.to(self.device)
            print(f"📉 Snapshot Nodes: {valid_node_mask.sum().item()} (Current Task Only)")

            # --- Model Specific Pre-computations ---
            is_ewc_mode = self.ewc_lambda > 0.0
            is_lwf_mode = self.lwf_alpha > 0.0
            is_replay_mode = self.replay_buffer.buffer_size_per_class > 0

            if self.config.train.model == 'hogrl':
                print(f">>> [HOGRL] Pre-computing graphs for Task {task_id+1}...")
                order = self.config.model.get('num_orders', 3)
                snapshot_data.adjs = self._precompute_high_order_graphs(
                    snapshot_data.edge_index, self.dataset.num_nodes, order=order
                )
            
            if self.config.train.model == 'consisgad':
                print(">>> [ConsisGAD] Pre-training LGA Module...")
                for lga_ep in range(20): 
                    ld, lg = self._train_lga_step(snapshot_data)

            # --- Epoch Loop ---
            for epoch in range(1, epochs_per_task + 1):
                global_step += 1
                self.model.train()
                self.optimizer.zero_grad()
                
                replay_idx = self.replay_buffer.get_buffer_indices().to(self.config.train.device)
                current_train_idx = torch.cat([task_train_idx, replay_idx])
                
                # --- GRAD MODEL SPECIAL FLOW ---
                if self.config.train.model == 'grad':
                    # A. Train Supervised GCL Encoder (Eq. 2)
                    z_gcl = self.model.forward_gcl(snapshot_data)
                    # Compute GCL Loss only on training nodes
                    gcl_labels = self.dataset.y[current_train_idx.cpu()].to(self.device)
                    loss_gcl = self._grad_gcl_loss(z_gcl[current_train_idx], gcl_labels)
                    
                    # B. Train Diffusion (Algorithm 1)
                    # Node Group Sampling
                    groups, perm_idx = self._node_group_sampling(snapshot_data.num_nodes, self.model.group_size)
                    
                    # 随机采样 64 个 Group 进行训练
                    num_sample_groups = 64
                    if len(groups) > num_sample_groups:
                        sampled_grp_idx = torch.randperm(len(groups))[:num_sample_groups]
                        batch_groups = [groups[i] for i in sampled_grp_idx]
                    else:
                        batch_groups = groups

                    adj_batch_list = []
                    for grp_nodes in batch_groups:
                        # 确保节点索引在 GPU 上以便进行 subgraph 操作
                        grp_nodes = grp_nodes.to(self.device)
                        
                        # 1. 提取子图边 (relabel_nodes=True 会自动把索引映射到 0~31)
                        sub_edge_index, _ = subgraph(
                            grp_nodes, 
                            snapshot_data.edge_index, 
                            relabel_nodes=True, 
                            num_nodes=snapshot_data.num_nodes
                        )
                        
                        # 2. 转为 Dense (32x32)
                        # max_num_nodes 确保维度固定为 group_size
                        dense_adj = to_dense_adj(sub_edge_index, max_num_nodes=self.model.group_size)[0]
                        adj_batch_list.append(dense_adj)
                    
                    # 堆叠为 Batch [B, 32, 32]
                    adj_batch = torch.stack(adj_batch_list)
                    
                    # Sample t
                    current_batch_size = adj_batch.size(0)
                    t = torch.randint(0, self.model.diff_steps, (current_batch_size,), device=self.device)
                    noise = torch.randn_like(adj_batch)
                    
                    # Diffusion Forward (Simplified linear schedule)
                    alpha_cumprod = torch.linspace(0.99, 0.01, self.model.diff_steps).to(self.device)
                    alpha_t = alpha_cumprod[t].view(-1, 1, 1)
                    
                    noisy_adj = torch.sqrt(alpha_t) * adj_batch + torch.sqrt(1 - alpha_t) * noise
                    noise_pred = self.model.denoise_net(noisy_adj, t)
                    loss_diff = F.mse_loss(noise_pred, noise)
                    
                    # C. Train Detector (Beta Wavelet)
                    # Note: We skip the slow generation process in training loop and train on identity augmentation or perturbation
                    # In true implementation, you would generate a new graph here.
                    out, _ = self.model(snapshot_data, generated_adj=None)
                    outputs = out.reshape((self.dataset.x.shape[0]))
                    
                    task_y = self.dataset.y[current_train_idx.cpu()].float().to(self.device).reshape(-1, 1)
                    loss_det = self.criterion(outputs[current_train_idx].reshape(-1, 1), task_y)
                    
                    # Total Loss (Multi-task)
                    total_loss = loss_det + 0.1 * loss_gcl + 0.1 * loss_diff

                # --- STANDARD FLOW FOR OTHER MODELS ---
                else:
                    if self.config.train.model == 'pmp':
                        pmp_mask = torch.zeros(self.dataset.num_nodes, dtype=torch.bool, device=self.device)
                        pmp_mask[current_train_idx] = True
                        self.dataset.pmp_mask = pmp_mask 

                    consis_view_outputs = None
                    if self.config.train.model == 'consisgad':
                        with torch.no_grad():
                            prob_adj = self.model.generator(snapshot_data.x, snapshot_data.edge_index)
                            adj_sample = self.model.generator.sample_adj(prob_adj)
                            consis_view_outputs = self.model.classifier(snapshot_data.x, snapshot_data.edge_index, adj_sample)

                    z_all = None; alpha_all = None; proj_features = None
                    out_res = self.model(snapshot_data)
                    
                    if isinstance(out_res, tuple):
                        if len(out_res) == 3: outputs, z_all, alpha_all = out_res
                        elif len(out_res) == 2: outputs, proj_features = out_res 
                        else: outputs = out_res[0]
                    else:
                        outputs = out_res
                    
                    outputs = outputs.reshape((self.dataset.x.shape[0]))
                    task_y = self.dataset.y[current_train_idx.cpu()].float().to(self.device).reshape(-1, 1)
                    
                    if self.config.train.model == 'gat_cobo':
                        sw = torch.ones_like(task_y); sw[task_y==1] = 10.0
                        task_loss = F.binary_cross_entropy_with_logits(
                            outputs[current_train_idx].double().reshape(-1, 1), task_y, weight=sw, pos_weight=self.criterion.pos_weight)
                    else:
                        task_loss = self.criterion(outputs[current_train_idx].reshape(-1, 1), task_y)
                    
                    grad_loss, bsl_loss, consis_loss, cl_loss = 0.0, 0.0, 0.0, 0.0
                    
                    # ConsisGAD Loss
                    if self.config.train.model == 'consisgad' and consis_view_outputs is not None:
                        all_idx = torch.arange(self.dataset.num_nodes, device=self.config.train.device)
                        is_not_train = ~torch.isin(all_idx, current_train_idx)
                        is_visible = valid_node_mask
                        is_current_task = (self.dataset.timesteps >= time_steps[0]) & (self.dataset.timesteps <= time_steps[-1])
                        unlabeled_mask = is_not_train & is_visible & is_current_task
                        if unlabeled_mask.sum() > 0:
                            p_orig = torch.sigmoid(outputs[unlabeled_mask])
                            p_aug = torch.sigmoid(consis_view_outputs.reshape(-1)[unlabeled_mask])
                            consis_loss = F.mse_loss(p_orig, p_aug)

                    # BSL Loss
                    if self.config.train.model == 'bsl' and z_all is not None:
                        alpha_train = alpha_all[current_train_idx]
                        y_train_long = self.dataset.y[current_train_idx].long()
                        l_attn = torch.mean(
                            (1 - y_train_long.float()) * F.relu(alpha_train[:, 1] - alpha_train[:, 2] + 0.2) + 
                            (y_train_long.float()) * F.relu(alpha_train[:, 2] - alpha_train[:, 1] + 0.2)
                        )
                        bsl_loss = 0.4 * l_attn 
                    
                    # CL Losses (LwF / EWC)
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
                    
                    total_loss = task_loss + cl_loss
                    if self.config.train.model == 'bsl': total_loss += bsl_loss
                    if self.config.train.model == 'consisgad': total_loss += 1.0 * consis_loss
                
                # --- Backward & Optimize ---
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step(total_loss)

                # ConsisGAD Auto-Tuning
                if (self.config.train.model == 'consisgad') and epoch % 5 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_out = self.model(snapshot_data) 
                        if isinstance(val_out, tuple): val_out = val_out[0]
                        val_preds = torch.sigmoid(val_out.reshape(-1)[task_valid_idx]).cpu().numpy()
                        val_labels = self.dataset.y[task_valid_idx].cpu().numpy()
                        metrics = self.compute_metrics(val_preds, val_labels)
                    self.model.train()
                    cw = self.criterion.pos_weight.item()
                    nw = cw
                    if metrics['recall'] < 0.7: nw = min(cw * 1.5, 25.0) 
                    elif metrics['precision'] < 0.3: nw = max(cw * 0.8, 1.0)
                    if nw != cw: 
                         self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([nw]).to(self.config.train.device).double())

            # --- Task End ---
            if is_ewc_mode: self._update_ewc_metrics(task_train_idx, snapshot_data)
            if is_lwf_mode:
                self.old_model = copy.deepcopy(self.model)
                self.old_model.to(self.config.train.device)
            if is_replay_mode:
                idx_cpu = task_train_idx.cpu()
                task_train_labels = self.dataset.y[idx_cpu].numpy() 
                self.replay_buffer.add_exemplars(idx_cpu.numpy(), task_train_labels)

            task_duration = time.time() - task_start_time
            del task_train_idx, task_valid_idx, snapshot_data 
            gc.collect(); torch.cuda.empty_cache()

            metrics_entry = self.evaluate_cl_metrics(task_id, task_duration)
            metrics_entry["cl_mode"] = "EWC" if is_ewc_mode else ("LwF" if is_lwf_mode else ("Replay" if is_replay_mode else "Naive"))
            self.aggregate_metrics_history.append(metrics_entry)
            gc.collect(); torch.cuda.empty_cache()

        print("All tasks trained.")
        total_time = time.time() - start_time_total
        print(f"Total time: {total_time:.2f}s")
        
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
