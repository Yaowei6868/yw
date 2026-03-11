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
    GATv2, HOGRL, CGNN, GradGNN, BSL, PMPModel, ConsisGAD, GraphSMOTE
)
from .datasets import EllipticDataset, EllipticPlusActorDataset, TFinanceDataset
from .buffer import ReplayBuffer, SubspacePrototypeBuffer

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

# ==========================================
# TASD-CL: BSL 参数语义角色 -> SSF λ 乘数
# 子空间路由参数(att_vec/att_bias)获最高约束，编码器参数获最低约束
# ==========================================
BSL_PARAM_GROUPS = {
    'att_vec':      3.0,   # 子空间路由 (最高): 定义欺诈特征归属，遗忘代价极高
    'att_bias':     3.0,
    'edge_decoder': 2.0,   # 边类型分类 (中等)
    'gnn_encoder':  0.5,   # 图编码器 (低): 图结构随任务变化，应允许更新
    'lin_in':       0.5,
    'classifier':   0.3,   # 最终分类器 (最低): 任务特定层
}

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
    "fraudgnn_rl": GATv2,
    "graphsmote": GraphSMOTE,
}

# 【修改点 1】: 将 tfinance 注册到数据集映射表中
datasets_map = {
    "elliptic": EllipticDataset,
    "elliptic_actor": EllipticPlusActorDataset,
    "tfinance": TFinanceDataset
}


class Trainer:
    # 初始化训练器
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
        
        # 【修改点 2】: 实例化数据集逻辑，增加 tfinance 分支并动态获取任务数
        if self.config.train.dataset == 'elliptic_actor':
            self.dataset_obj = EllipticPlusActorDataset(root='data/elliptic++actor')
            self.dataset = self.dataset_obj[0]
        elif self.config.train.dataset == 'tfinance':
            # 动态获取 yaml 中的任务数量 (task_schedule 的长度)，默认 10
            num_tasks = len(self.config.train.task_schedule) if hasattr(self.config.train, 'task_schedule') else 10
            self.dataset_obj = TFinanceDataset(root='data/tfinance', num_tasks=num_tasks)
            self.dataset = self.dataset_obj[0]
        else:
            # 兼容旧逻辑 (原版 Elliptic)
            self.dataset_obj = datasets_map[self.config.train.dataset](config.dataset)
            self.dataset = self.dataset_obj.pyg_dataset().to(self.device)
            
        # 统一转为 float32，避免与模型权重的 dtype 不匹配
        self.dataset.x = self.dataset.x.float()

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
        # [TASD-CL 初始化] ---------------------------------------------------
        self.spc_buffer = None
        if self.config.train.model == 'bsl':
            sub_dim = config.model.hidden_dim // 3
            self.spc_buffer = SubspacePrototypeBuffer(sub_dim=sub_dim)
        self.spc_lambda = config.train.get('spc_lambda', 0.0)
        self.scd_lambda = config.train.get('scd_lambda', 0.0)
        self.scd_tau    = config.train.get('scd_tau', 0.5)
        # --------------------------------------------------------------------
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
        # [省略无关更改的代码，保持与你原版完全一致...]
        x_nor_train = x_nor[train_idx]
        x_abnor_train = x_abnor[train_idx]
        nor_norm = F.normalize(x_nor_train, p=2, dim=1)
        abnor_norm = F.normalize(x_abnor_train, p=2, dim=1)
        cos_sim = (nor_norm * abnor_norm).sum(dim=1)
        loss_csd = (cos_sim ** 2).mean()
        
        y_train = y[train_idx]
        norm_n = torch.norm(x_nor_train, p=2, dim=1)
        norm_a = torch.norm(x_abnor_train, p=2, dim=1)
        margin = 0.5
        loss_0 = F.relu(norm_a - norm_n + margin)
        loss_1 = F.relu(norm_n - norm_a + margin)
        loss_consis = torch.mean(torch.where(y_train == 0, loss_0, loss_1))
        
        return loss_csd, loss_consis

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

    def _update_ewc_metrics(self, task_train_idx: torch.Tensor, dataset):
        if self.ewc_lambda <= 0.0: return     
        print("   [EWC] Updating Fisher Matrix...")
        self.model.eval() 
        self.optimizer.zero_grad()      
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
        idx_cpu = task_train_idx.cpu()
        task_y = self.dataset.y[idx_cpu].float().to(self.device).reshape(-1, 1)    
        pred = outputs[task_train_idx].reshape(-1, 1)
        loss = self.criterion(pred, task_y)
        loss.backward() 
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if name not in self.ewc_fisher: 
                    self.ewc_fisher[name] = torch.zeros_like(param.data).to(param.device)
                self.ewc_fisher[name].data += param.grad.data.pow(2)
                self.ewc_params[name] = param.data.clone()
    
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
    
    def _node_group_sampling(self, num_nodes, group_size=32):
        indices = torch.randperm(num_nodes)
        num_groups = num_nodes // group_size
        groups = []
        for i in range(num_groups):
            groups.append(indices[i*group_size : (i+1)*group_size])
        return groups, indices 
    
    def _grad_gcl_loss(self, z, y, temperature=0.2):
        batch_size = 1024
        if z.size(0) > batch_size:
            idx = torch.randperm(z.size(0))[:batch_size]
            z = z[idx]
            y = y[idx]
            
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / temperature
        exp_sim = torch.exp(sim)
        mask = torch.eq(y.view(-1, 1), y.view(-1, 1)).float().to(z.device)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0)).to(z.device)
        mask = mask * logits_mask
        exp_sim = exp_sim * logits_mask
        denominator = exp_sim.sum(dim=1, keepdim=True)
        log_prob = sim - torch.log(denominator + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        return -mean_log_prob_pos.mean()
    
    def _get_edge_types(self, edge_index, y):
        src, dst = edge_index
        y_src = y[src]
        y_dst = y[dst]
        mask_labeled = (y_src != -1) & (y_dst != -1)
        src_l, dst_l = src[mask_labeled], dst[mask_labeled]
        y_s, y_d = y_src[mask_labeled], y_dst[mask_labeled]
        edge_types = torch.zeros_like(y_s).long() 
        mask_aa = (y_s == 1) & (y_d == 1)
        edge_types[mask_aa] = 1
        mask_nn = (y_s == 0) & (y_d == 0)
        edge_types[mask_nn] = 2
        mask_na = (y_s != y_d)
        edge_types[mask_na] = 0
        return src_l, dst_l, edge_types
    
    def _compute_bsl_full_loss(self, model, data, outputs, z_all, alpha_all, train_idx, valid_node_mask):
        device = self.device
        y = data.y
        src_l, dst_l, edge_types = self._get_edge_types(data.edge_index, y)
        if len(src_l) > 0:
            if len(src_l) > 2048:
                idx = torch.randperm(len(src_l))[:2048]
                src_l, dst_l, edge_types = src_l[idx], dst_l[idx], edge_types[idx]
            z_src = z_all[src_l]
            z_dst = z_all[dst_l]
            z_src_parts = model.get_sub_features(z_src)
            z_dst_parts = model.get_sub_features(z_dst)
            link_losses = []
            for k in range(3): 
                feat_cat = torch.cat([z_src_parts[k], z_dst_parts[k]], dim=1)
                logits = model.edge_decoder(feat_cat).squeeze()
                target = (edge_types == k).float()
                link_losses.append(F.binary_cross_entropy_with_logits(logits, target))
            l_link = sum(link_losses)
        else:
            l_link = torch.tensor(0.0).to(device)

        alpha_train = alpha_all[train_idx]
        y_train = y[train_idx].float()
        margin = 0.2
        loss_norm = F.relu(alpha_train[:, 1] - alpha_train[:, 2] + margin) 
        loss_fraud = F.relu(alpha_train[:, 2] - alpha_train[:, 1] + margin) 
        l_attn = torch.mean((1 - y_train) * loss_norm + y_train * loss_fraud)
        l_d = l_link + l_attn

        pos_mask = (y[train_idx] == 1)
        neg_mask = (y[train_idx] == 0)
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return l_d, torch.tensor(0.0).to(device)
            
        idx_n = train_idx[neg_mask][torch.randint(0, neg_mask.sum(), (1,))]
        idx_a = train_idx[pos_mask][torch.randint(0, pos_mask.sum(), (1,))]
        z_n = z_all[idx_n] 
        z_a = z_all[idx_a] 
        
        all_indices = torch.where(valid_node_mask)[0]
        unlabeled_indices = all_indices[torch.randperm(len(all_indices))[:256]] 
        z_u = z_all[unlabeled_indices] 
        
        def reconstruct(z_parts_list):
            new_alpha = model.get_attention_weights(z_parts_list)
            z_w = torch.cat([
                z_parts_list[0] * new_alpha[:, 0:1],
                z_parts_list[1] * new_alpha[:, 1:2],
                z_parts_list[2] * new_alpha[:, 2:3]
            ], dim=1)
            return model.classifier(z_w)

        zu_parts = model.get_sub_features(z_u) 
        zn_parts = model.get_sub_features(z_n)
        za_parts = model.get_sub_features(z_a)
        
        zn_parts_exp = [p.expand(z_u.size(0), -1) for p in zn_parts]
        za_parts_exp = [p.expand(z_u.size(0), -1) for p in za_parts]

        z_nw_parts = [zn_parts_exp[0], zu_parts[1], zu_parts[2]]
        z_aw_parts = [za_parts_exp[0], zu_parts[1], zu_parts[2]]
        pred_nw = reconstruct(z_nw_parts)
        pred_aw = reconstruct(z_aw_parts)
        l_con = F.mse_loss(torch.sigmoid(pred_nw), torch.sigmoid(pred_aw))
        
        z_ns_parts = [zu_parts[0], zn_parts_exp[1], zn_parts_exp[2]]
        z_as_parts = [zu_parts[0], za_parts_exp[1], za_parts_exp[2]]
        pred_ns = reconstruct(z_ns_parts)
        pred_as = reconstruct(z_as_parts)
        l_incon = -F.mse_loss(torch.sigmoid(pred_ns), torch.sigmoid(pred_as))
        
        l_bsl = l_con + 0.5 * l_incon 
        return l_d, l_bsl

    def _consisgad_loss_consistency(self, p_orig, p_aug):
        prob_orig = torch.sigmoid(p_orig)
        prob_aug = torch.sigmoid(p_aug)
        loss = F.mse_loss(prob_aug, prob_orig.detach())
        return loss

    def _consisgad_loss_diversity(self, h_orig, h_aug):
        dist = torch.norm(h_orig.detach() - h_aug, p=2, dim=1).mean()
        return -dist 
    
    def _train_lga_step(self, snapshot_data):
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

    # =========================================================================
    # TASD-CL 专用方法
    # =========================================================================

    def _get_ssf_lambda(self, param_name: str) -> float:
        """
        [TASD-CL] Component A: Subspace-Stratified Fisher (SSF)
        根据 BSL 参数的语义角色返回 λ 乘数。
        子空间路由参数获最高约束，避免欺诈子空间在任务切换时整体坍塌。
        """
        for pattern, multiplier in BSL_PARAM_GROUPS.items():
            if pattern in param_name:
                return multiplier
        return 1.0

    def _compute_scd_loss(
        self,
        snapshot_data,
        new_z: torch.Tensor,
        new_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        [TASD-CL] Component C: Subspace-Conditioned Distillation (SCD)

        在 BSL 三个解耦子空间上进行置信度过滤的知识蒸馏。
        仅蒸馏旧模型对某子空间赋予高置信度 (alpha_k > scd_tau) 的节点，
        规避不平衡数据集上旧模型不可靠预测的负迁移。

        相比标准 LwF 只蒸馏 1 维输出，SCD 保留了 3×sub_dim 维子空间几何。
        Z_aa（欺诈感知子空间）赋予最高蒸馏权重。
        """
        if self.old_model is None:
            return torch.tensor(0.0, device=self.device)

        self.old_model.eval()
        with torch.no_grad():
            _, old_z, old_alpha = self.old_model(snapshot_data, return_stats=True)

        old_parts = self.model.get_sub_features(old_z.detach())
        new_parts = self.model.get_sub_features(new_z)

        # Z_na:1.0, Z_aa:2.0, Z_nn:1.0 — 欺诈子空间权重最高
        subspace_weights = [1.0, 2.0, 1.0]
        loss_scd = torch.tensor(0.0, device=self.device)

        for k in range(3):
            # 只蒸馏旧模型对第 k 个子空间高置信度的节点
            confidence_mask = (old_alpha[:, k] > self.scd_tau).detach()
            if confidence_mask.sum() < 2:
                continue
            loss_scd = loss_scd + subspace_weights[k] * F.mse_loss(
                new_parts[k][confidence_mask],
                old_parts[k][confidence_mask].detach(),
            )
        return loss_scd

    def _update_spc_prototypes(
        self,
        task_id: int,
        snapshot_data,
        task_train_idx: torch.Tensor,
    ):
        """
        [TASD-CL] Component B: Subspace Prototype Condensation (SPC) 更新
        在任务结束后提取 BSL 三子空间的类原型并存入 SPC 缓冲区。
        存储 Gaussian 原型 (μ, σ) 而非原始节点索引，实现图结构无关的回放。
        """
        if self.spc_buffer is None or self.spc_lambda <= 0.0:
            return

        self.model.eval()
        with torch.no_grad():
            _, z_all, _ = self.model(snapshot_data, return_stats=True)

        z_train = z_all[task_train_idx]
        z_parts_train = list(self.model.get_sub_features(z_train))
        y_train = self.dataset.y[task_train_idx.cpu()].to(self.device)

        self.spc_buffer.add_task_prototypes(task_id, z_parts_train, y_train)

        fraud_n  = (y_train == 1).sum().item()
        normal_n = (y_train == 0).sum().item()
        print(f"   [SPC] Task {task_id+1} 子空间原型已更新 | 欺诈: {fraud_n}, 正常: {normal_n}")
        self.model.train()

    def evaluate_cl_metrics(self, current_task_id, task_duration):
        print(f"\n--- [CL Evaluation] Evaluating on all seen tasks (0 to {current_task_id}) ---")
        self.model.eval()
        row_metrics = {
            "f1": [], "recall": [], "precision": [],
            "auc_roc": [], "auc_pr": [], "g_mean": [], 
            "avg_cost": [] 
        }

        for t_id in range(current_task_id + 1):
            if t_id not in self.task_valid_indices_map:
                for k in row_metrics: row_metrics[k].append(0.0)
                continue

            task_start = self.task_schedule[t_id][0]
            task_end = self.task_schedule[t_id][-1]
            eval_mask = (self.dataset.timesteps >= task_start) & (self.dataset.timesteps <= task_end)
            
            row, col = self.dataset.edge_index
            edge_mask = eval_mask[row] & eval_mask[col]
            
            snapshot_data = copy.copy(self.dataset)
            snapshot_data.edge_index = self.dataset.edge_index[:, edge_mask]
            snapshot_data = snapshot_data.to(self.device) 

            with torch.no_grad():
                if self.config.train.model == 'hogrl':
                    order = self.config.model.get('num_orders', 3)
                    snapshot_data.adjs = self._precompute_high_order_graphs(
                        snapshot_data.edge_index, self.dataset.num_nodes, order=order
                    )
                
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
                labels_all = self.dataset.y.cpu().numpy()

                valid_idx = self.task_valid_indices_map[t_id]
                t_preds = probs[valid_idx]
                t_labels = labels_all[valid_idx]
                
                res = self.compute_metrics(t_preds, t_labels, threshold=0.15)
                
                if hasattr(self, 'f1_matrix'):
                    self.f1_matrix[current_task_id, t_id] = res['f1']
                
                for k in row_metrics:
                    if k in res:
                        row_metrics[k].append(res[k])

            del snapshot_data, outputs
            torch.cuda.empty_cache()

        avg_metrics = {k: np.mean(v) for k, v in row_metrics.items()}
        avg_forgetting = 0.0
        avg_bwt = 0.0
        
        if current_task_id > 0 and hasattr(self, 'f1_matrix'):
            forgetting_sum = 0.0
            bwt_sum = 0.0
            for j in range(current_task_id): 
                history_best = self.f1_matrix[:current_task_id, j].max()
                current_score = self.f1_matrix[current_task_id, j]
                forgetting_sum += (history_best - current_score)
                original_score = self.f1_matrix[j, j] 
                bwt_sum += (current_score - original_score)
            
            avg_forgetting = forgetting_sum / current_task_id
            avg_bwt = bwt_sum / current_task_id

        print(f"*** CL Metrics @ Task {current_task_id+1} ***")
        print(f"  [Accuracy]  Avg F1: {avg_metrics['f1']:.4f} | AUC-ROC: {avg_metrics['auc_roc']:.4f} | G-Mean: {avg_metrics['g_mean']:.4f}")
        print(f"  [Stability] Avg Forgetting (↓): {avg_forgetting:.4f} | Avg BWT (↑): {avg_bwt:.4f}")
        
        if hasattr(self, 'f1_matrix'):
            current_row = self.f1_matrix[current_task_id, :current_task_id+1]
            print(f"  > Matrix Row:  {np.round(current_row, 4)}")
        
        for k, v in avg_metrics.items():
            self.tensorboard.add_scalar(f"CL/Avg_{k}", v, current_task_id + 1)
        self.tensorboard.add_scalar("CL/Avg_Forgetting", avg_forgetting, current_task_id + 1)
        self.tensorboard.add_scalar("CL/Avg_BWT", avg_bwt, current_task_id + 1)

        result_entry = {
            "task_id": current_task_id + 1,
            "time_cost": task_duration,
            "avg_forgetting": avg_forgetting,
            "avg_bwt": avg_bwt,
            **{f"avg_{k}": v for k, v in avg_metrics.items()}
        }
        return result_entry

    def train(self):
        epochs_per_task = self.config.train.get('num_epochs_per_task', 50) 
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
            
            y_curr = self.dataset.y[task_train_idx.cpu()]
            num_pos = (y_curr == 1).sum().item()
            num_neg = (y_curr == 0).sum().item()
            raw_ratio = num_neg / num_pos if num_pos > 0 else 1.0
            
            if task_id == 0:
                clip_cap = 20.0 
            else:
                clip_cap = 10.0 
            
            clipped_weight = max(min(raw_ratio, clip_cap), 1.0) 
            print(f"🔧 Task {task_id+1} 动态权重: {raw_ratio:.2f} -> 截断为: {clipped_weight:.2f}")
            
            dynamic_alpha = clipped_weight / (1.0 + clipped_weight)
            print(f"   -> Focal Loss Alpha: {dynamic_alpha:.4f}")
            
            self.criterion = BinaryFocalLoss(alpha=dynamic_alpha, gamma=2.0).to(self.device)

            task_start_t, task_end_t = time_steps[0], time_steps[-1]
            valid_node_mask = (self.dataset.timesteps >= task_start_t) & (self.dataset.timesteps <= task_end_t)
            row, col = self.dataset.edge_index
            edge_mask = valid_node_mask[row] & valid_node_mask[col]
            
            snapshot_data = copy.copy(self.dataset)
            snapshot_data.edge_index = self.dataset.edge_index[:, edge_mask]
            snapshot_data = snapshot_data.to(self.device) 
            print(f"📉 Snapshot 节点数: {valid_node_mask.sum().item()} (仅当前任务)")

            is_ewc_mode = self.ewc_lambda > 0.0
            is_lwf_mode = self.lwf_alpha > 0.0
            is_replay_mode = self.replay_buffer.buffer_size_per_class > 0

            if self.config.train.model == 'hogrl':
                print(f">>> [HOGRL] 正在为 Task {task_id+1} 预计算高阶图...")
                order = self.config.model.get('num_orders', 3)
                snapshot_data.adjs = self._precompute_high_order_graphs(
                    snapshot_data.edge_index, self.dataset.num_nodes, order=order
                )
            
            if self.config.train.model == 'consisgad' and hasattr(self.model, 'discriminator'):
                print(">>> [ConsisGAD] 预训练 LGA 模块...")
                for lga_ep in range(20): 
                    ld, lg = self._train_lga_step(snapshot_data)

            for epoch in range(1, epochs_per_task + 1):
                global_step += 1
                self.model.train()
                self.optimizer.zero_grad()
                
                replay_idx = self.replay_buffer.get_buffer_indices().to(self.config.train.device)
                current_train_idx = torch.cat([task_train_idx, replay_idx])
                
                if self.config.train.model == 'grad':
                    z_gcl = self.model.forward_gcl(snapshot_data)
                    gcl_labels = self.dataset.y[current_train_idx.cpu()].to(self.device)
                    loss_gcl = self._grad_gcl_loss(z_gcl[current_train_idx], gcl_labels)
                    
                    groups, perm_idx = self._node_group_sampling(snapshot_data.num_nodes, self.model.group_size)
                    
                    num_sample_groups = 64
                    if len(groups) > num_sample_groups:
                        sampled_grp_idx = torch.randperm(len(groups))[:num_sample_groups]
                        batch_groups = [groups[i] for i in sampled_grp_idx]
                    else:
                        batch_groups = groups

                    adj_batch_list = []
                    for grp_nodes in batch_groups:
                        grp_nodes = grp_nodes.to(self.device)
                        sub_edge_index, _ = subgraph(
                            grp_nodes, 
                            snapshot_data.edge_index, 
                            relabel_nodes=True, 
                            num_nodes=snapshot_data.num_nodes
                        )
                        dense_adj = to_dense_adj(sub_edge_index, max_num_nodes=self.model.group_size)[0]
                        adj_batch_list.append(dense_adj)
                    
                    adj_batch = torch.stack(adj_batch_list)
                    
                    current_batch_size = adj_batch.size(0)
                    t = torch.randint(0, self.model.diff_steps, (current_batch_size,), device=self.device)
                    noise = torch.randn_like(adj_batch)
                    
                    alpha_cumprod = torch.linspace(0.99, 0.01, self.model.diff_steps).to(self.device)
                    alpha_t = alpha_cumprod[t].view(-1, 1, 1)
                    noisy_adj = torch.sqrt(alpha_t) * adj_batch + torch.sqrt(1 - alpha_t) * noise
                    
                    noise_pred = self.model.denoise_net(noisy_adj, t)
                    loss_diff = F.mse_loss(noise_pred, noise)
                    
                    out, _ = self.model(snapshot_data, generated_adj=None)
                    outputs = out.reshape((self.dataset.x.shape[0]))
                    
                    task_y = self.dataset.y[current_train_idx.cpu()].float().to(self.device).reshape(-1, 1)
                    loss_det = self.criterion(outputs[current_train_idx].reshape(-1, 1), task_y)
                    
                    total_loss = loss_det + 0.1 * loss_gcl + 0.1 * loss_diff
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step(total_loss)

                elif self.config.train.model == 'consisgad':
                    if not hasattr(self, 'aug_optimizer'):
                        self.aug_optimizer = optim.Adam(
                            self.model.augmentor.parameters(), 
                            lr=self.config.train.lr, 
                            weight_decay=self.config.train.weight_decay
                        )
                    
                    out_real, out_aug, h_real, h_aug = self.model(snapshot_data, augment=True)
                    
                    outputs = out_real.reshape((self.dataset.x.shape[0]))
                    out_aug = out_aug.reshape((self.dataset.x.shape[0]))
                    
                    warmup_epochs = 10 
                    is_warmup = epoch <= warmup_epochs
                    
                    mask_hq = torch.zeros(outputs.shape[0], dtype=torch.bool, device=self.device)
                    
                    if not is_warmup:
                        probs = torch.sigmoid(outputs).detach() 
                        all_indices = torch.arange(snapshot_data.num_nodes, device=self.device)
                        is_unlabeled = ~torch.isin(all_indices, current_train_idx)
                        
                        tau_n, tau_a = 0.70, 0.60
                        mask_hq = ((probs < (1 - tau_n)) | (probs > tau_a)) & is_unlabeled
                    
                    if not is_warmup and mask_hq.sum() > 0:
                        l_consist = self._consisgad_loss_consistency(outputs[mask_hq].detach(), out_aug[mask_hq])
                        l_diver = self._consisgad_loss_diversity(h_real[mask_hq].detach(), h_aug[mask_hq])
                        loss_aug = l_consist + l_diver
                        
                        self.aug_optimizer.zero_grad()
                        loss_aug.backward() 
                        self.aug_optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    out_real_new = self.model(snapshot_data, augment=False)
                    outputs_new = out_real_new.reshape((self.dataset.x.shape[0]))
                    
                    task_y = self.dataset.y[current_train_idx.cpu()].float().to(self.device).reshape(-1, 1)
                    loss_sup = self.criterion(outputs_new[current_train_idx].reshape(-1, 1), task_y)
                    
                    loss_consist_gnn = 0.0
                    if not is_warmup and mask_hq.sum() > 0:
                        pseudo_labels = (probs[mask_hq] > 0.5).float().detach()
                        h_aug_fixed = h_aug.detach() 
                        
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

                else:
                    if self.config.train.model == 'pmp':
                        pmp_mask = torch.zeros(self.dataset.num_nodes, dtype=torch.bool, device=self.device)
                        pmp_mask[current_train_idx] = True
                        self.dataset.pmp_mask = pmp_mask 

                    z_all = None; alpha_all = None
                    
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
                    
                    task_loss = self.criterion(outputs[current_train_idx].reshape(-1, 1), task_y)
                    
                    bsl_loss, cl_loss, cgnn_loss = 0.0, 0.0, 0.0
                    spc_loss = torch.tensor(0.0, device=self.device)  # [TASD-CL]

                    if self.config.train.model == 'cgnn':
                        w_csd = self.config.train.get('cgnn_lambda', 0.1)
                        w_consist = self.config.train.get('cgnn_beta', 0.1)
                        l_csd, l_consist = self._cgnn_loss(x_nor, x_abnor, self.dataset.y.to(self.device), current_train_idx)
                        cgnn_loss = w_csd * l_csd + w_consist * l_consist

                    if self.config.train.model == 'bsl' and z_all is not None:
                        l_d, l_bsl_term = self._compute_bsl_full_loss(
                            self.model, snapshot_data, outputs, z_all, alpha_all,
                            current_train_idx, valid_node_mask
                        )
                        bsl_loss = 0.4 * l_d + 0.8 * l_bsl_term

                        # [TASD-CL] Component B: 子空间原型凝缩回放 (SPC)
                        # 从历史任务的子空间原型中采样合成节点，直接通过 BSL 分类器计算回放损失。
                        # 绕过 GNN 前向传播，完全图结构无关。
                        if self.spc_lambda > 0 and self.spc_buffer is not None \
                                and self.spc_buffer.has_prototypes():
                            n_per_cls = self.config.train.get('spc_n_samples', 32)
                            z_rep, y_rep = self.spc_buffer.sample_prototypes(n_per_cls, self.device)
                            if z_rep is not None:
                                z_rep_parts = self.model.get_sub_features(z_rep)
                                alpha_rep   = self.model.get_attention_weights(z_rep_parts)
                                z_rep_w = torch.cat([
                                    z_rep_parts[0] * alpha_rep[:, 0:1],
                                    z_rep_parts[1] * alpha_rep[:, 1:2],
                                    z_rep_parts[2] * alpha_rep[:, 2:3],
                                ], dim=1)
                                out_rep  = self.model.classifier(z_rep_w)
                                spc_loss = F.binary_cross_entropy_with_logits(
                                    out_rep, (y_rep == 1).float().unsqueeze(1)
                                )

                    if is_lwf_mode and task_id > 0:
                        self.old_model.eval()
                        with torch.no_grad():
                            old_out = self.old_model(snapshot_data)
                            if isinstance(old_out, tuple): old_out = old_out[0]
                        cl_loss += self.lwf_alpha * self._distillation_loss(
                            torch.sigmoid(outputs[current_train_idx]),
                            torch.sigmoid(old_out.reshape(-1)[current_train_idx]))

                    # [TASD-CL] Component C: 置信度过滤子空间蒸馏 (SCD)
                    # 在 BSL 三子空间层面蒸馏，只蒸馏旧模型高置信度区域，避免不平衡偏差。
                    if self.scd_lambda > 0 and self.config.train.model == 'bsl' \
                            and task_id > 0 and self.old_model is not None \
                            and z_all is not None:
                        loss_scd = self._compute_scd_loss(snapshot_data, z_all, alpha_all)
                        cl_loss = cl_loss + self.scd_lambda * loss_scd

                    if is_ewc_mode and task_id > 0:
                        ewc_term = 0.0
                        for name, param in self.model.named_parameters():
                            if name in self.ewc_fisher:
                                # [TASD-CL] Component A: SSF - BSL 使用语义角色 λ 乘数
                                # att_vec/att_bias 约束最强，gnn_encoder/classifier 约束最弱
                                role_mult = self._get_ssf_lambda(name) \
                                    if self.config.train.model == 'bsl' else 1.0
                                ewc_term += role_mult * (
                                    self.ewc_fisher[name].to(param.device) *
                                    (param - self.ewc_params[name].to(param.device)).pow(2)
                                ).sum()
                        cl_loss += self.ewc_lambda * ewc_term

                    total_loss = task_loss + cl_loss + bsl_loss + cgnn_loss \
                                 + self.spc_lambda * spc_loss
                
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step(total_loss)

            if is_ewc_mode: self._update_ewc_metrics(task_train_idx, snapshot_data)
            if is_lwf_mode:
                self.old_model = copy.deepcopy(self.model)
                self.old_model.to(self.config.train.device)
            if is_replay_mode:
                idx_cpu = task_train_idx.cpu()
                task_train_labels = self.dataset.y[idx_cpu].numpy()
                self.replay_buffer.add_exemplars(idx_cpu.numpy(), task_train_labels)

            # [TASD-CL] Component B: 任务结束后更新 SPC 子空间原型库
            if self.config.train.model == 'bsl' and self.spc_lambda > 0:
                self._update_spc_prototypes(task_id, snapshot_data, task_train_idx)

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

    # 【修改点 3】: 增强 test 方法的兼容性
    def test(self, dataset=None, labeled_only=False, threshold=0.5):
        dataset = dataset or self.dataset
        self.model.eval()
        out_res = self.model(dataset)
        if isinstance(out_res, tuple): out_res = out_res[0]
        outputs = out_res.reshape((dataset.x.shape[0]))
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        
        # 安全处理: 如果新数据集没有明确的 test_idx (如 tfinance)，直接返回全图预测
        if hasattr(dataset, 'test_idx'):
            preds = probs if labeled_only else probs[dataset.test_idx]
        else:
            preds = probs
            
        return preds, preds > threshold

    def save(self, file_name):
        file_name = f"{file_name}.pt" if ".pt" not in file_name else file_name
        save_path = os.path.join(self.config.train.save_dir, file_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)