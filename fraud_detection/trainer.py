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

# 引入所有模型
from .models import (
    GAT, GCN, GIN, GraphSAGE, STAGNN, EvolveGCN, TGN, MLP, 
    GATv2, HOGRL, CGNN, GradGNN, BSL, PMPModel, ConsisGAD
)
from .datasets import EllipticDataset,EllipticPlusActorDataset
from .buffer import ReplayBuffer 

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
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
        # 实例化数据集 (适配 Elliptic++ Actor)
        if self.config.train.dataset == 'elliptic_actor':
            self.dataset_obj = EllipticPlusActorDataset(root='data/elliptic++actor')
            self.dataset = self.dataset_obj[0].to(self.device)
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
        
        # num_pos = (y_valid == 1).sum().item()
        # num_neg = (y_valid == 0).sum().item()
        # # 1. 计算原始比例
        # if num_pos > 0:
        #     raw_ratio = num_neg / num_pos
        # else:
        #     raw_ratio = 1.0  # 防止除以零
        # # 2. 【关键】加上安全锁 (Clip)，防止权重过大导致模型发疯
        # # 建议上限设为 5.0 或 6.0。对于 Naive 基线，保守一点更好。
        # self.current_pos_weight = min(raw_ratio, 5.0) 
        # self.current_pos_weight = max(self.current_pos_weight, 1.0) # 至少是 1.0
        # print(f"原始比例: {raw_ratio:.2f} -> 截断后权重: {self.current_pos_weight:.2f}")

        # self.criterion = nn.BCEWithLogitsLoss(
        #     pos_weight=torch.tensor([clipped_weight]).to(self.device)
        #     )

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

    def _distillation_loss(self, student_output, teacher_output):
        prob_s = student_output.squeeze().clamp(min=1e-7, max=1-1e-7)
        prob_t = teacher_output.detach().squeeze().clamp(min=1e-7, max=1-1e-7) 
        kl_div = prob_t * (torch.log(prob_t) - torch.log(prob_s)) + \
                 (1 - prob_t) * (torch.log(1 - prob_t) - torch.log(1 - prob_s))
        return kl_div.mean()

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

    def _update_ewc_metrics(self, task_train_idx: torch.Tensor):
        if self.ewc_lambda <= 0.0: return
        self.model.eval() 
        self.optimizer.zero_grad()
        out_res = self.model(self.dataset)
        outputs = out_res[0] if isinstance(out_res, tuple) else out_res
        outputs = outputs.reshape((-1,))
        task_y = self.dataset.y[task_train_idx].double().reshape(-1, 1)
        loss = self.criterion(outputs[task_train_idx].double().reshape(-1, 1), task_y)
        loss.backward() 
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if name not in self.ewc_fisher: self.ewc_fisher[name] = torch.zeros_like(param.data).to(param.device)
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
        print(f"Pre-computing high-order graphs up to order {order}...")
        device = self.device
        val = torch.ones(edge_index.size(1), dtype=torch.double)
        adj_coo = torch.sparse_coo_tensor(edge_index.cpu(), val, (num_nodes, num_nodes)).coalesce()
        adj = adj_coo.to_sparse_csr().to(device)
        adjs = [adj] 
        current_adj = adj
        for k in range(2, order + 1):
            try:
                current_adj = torch.matmul(current_adj, adj)
                adjs.append(current_adj)
                print(f"  - Order {k} graph computed.")
            except Exception as e:
                print(f"Error computing order {k} graph: {e}"); break
        return adjs

    def _get_bsl_augmentation(self, z_unlabeled, z_n, z_a, model):
        zu_na, zu_aa, zu_nn = model.get_sub_features(z_unlabeled) 
        zn_na, zn_aa, zn_nn = model.get_sub_features(z_n.unsqueeze(0))
        za_na, za_aa, za_nn = model.get_sub_features(z_a.unsqueeze(0))
        z_nw = torch.cat([zn_na.expand_as(zu_na), zu_aa, zu_nn], dim=1)
        z_aw = torch.cat([za_na.expand_as(zu_na), zu_aa, zu_nn], dim=1)
        z_ns = torch.cat([zu_na, zn_aa.expand_as(zu_aa), zn_nn.expand_as(zu_nn)], dim=1)
        z_as = torch.cat([zu_na, za_aa.expand_as(zu_aa), za_nn.expand_as(zu_nn)], dim=1)
        return z_nw, z_aw, z_ns, z_as

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
        
        # 初始化记录列表 (对应 compute_metrics 返回的 key)
        row_metrics = {
            "precision": [], "recall": [], "f1": [], 
            "macro_recall": [], "macro_f1": [], 
            "g_mean": [], "auc_roc": [], "auc_pr": [], 
            "total_cost": [], "avg_cost": []
        }

        with torch.no_grad():
            # Snapshot 构建
            current_max_step = self.task_schedule[current_task_id][-1]
            valid_node_mask = self.dataset.timesteps <= current_max_step
            row, col = self.dataset.edge_index
            edge_mask = valid_node_mask[row] & valid_node_mask[col]
            snapshot_data = copy.copy(self.dataset)
            snapshot_data.edge_index = self.dataset.edge_index[:, edge_mask]

            if self.config.train.model == 'hogrl':
                order = self.config.model.get('num_orders', 3)
                snapshot_data.adjs = self._precompute_high_order_graphs(
                    snapshot_data.edge_index, self.dataset.num_nodes, order=order
                )

            out_res = self.model(snapshot_data)
            if isinstance(out_res, tuple): outputs = out_res[0] 
            else: outputs = out_res
            outputs = outputs.reshape((self.dataset.x.shape[0]))
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            labels_all = self.dataset.y.detach().cpu().numpy()

            # 回测历史任务
            for t_id in range(current_task_id + 1):
                if t_id not in self.task_valid_indices_map:
                    # 缺失数据填 0
                    for k in row_metrics: row_metrics[k].append(0.0)
                    continue
                
                valid_idx = self.task_valid_indices_map[t_id]
                t_preds = probs[valid_idx]
                t_labels = labels_all[valid_idx]
                
                # 计算指标
                res = self.compute_metrics(t_preds, t_labels)
                
                # 存入列表
                for k in row_metrics:
                    row_metrics[k].append(res[k])
                
                if t_id == current_task_id:
                    current_task_metrics = res

        # 计算系统级平均指标 (Average over all seen tasks)
        avg_metrics = {k: np.mean(v) for k, v in row_metrics.items()}
        
        # 打印 (仅打印指标，不打印 Task Time)
        print(f"*** CL Metrics @ Task {current_task_id+1} ***")
        print(f"  [Binary Metrics (Fraud Class)]")
        print(f"  > Avg F1:           {avg_metrics['f1']:.4f}")
        print(f"  > Avg Precision:    {avg_metrics['precision']:.4f}")
        print(f"  > Avg Recall:       {avg_metrics['recall']:.4f}")
        
        print(f"  [Macro / Balanced Metrics]")
        print(f"  > Avg Macro F1:     {avg_metrics['macro_f1']:.4f}")
        print(f"  > Avg Macro Recall: {avg_metrics['macro_recall']:.4f}")
        print(f"  > Avg G-Mean:       {avg_metrics['g_mean']:.4f}")
        
        print(f"  [Ranking Metrics]")
        print(f"  > Avg AUC-ROC:      {avg_metrics['auc_roc']:.4f}")
        print(f"  > Avg AUC-PR:       {avg_metrics['auc_pr']:.4f}")
        
        print(f"  [Financial Cost]")
        print(f"  > Avg Total Cost:   {avg_metrics['total_cost']:.2f}")
        print(f"  > Avg Avg Cost:     {avg_metrics['avg_cost']:.4f}")
        
        # Tensorboard
        for k, v in avg_metrics.items():
            self.tensorboard.add_scalar(f"CL/Avg_{k}", v, current_task_id + 1)

        # 返回结果 (展平字典以便存 CSV)
        result_entry = {
            "task_id": current_task_id + 1,
            "time_cost": task_duration, # 仅保存，不打印
            # 添加所有 Avg 指标
            **{f"avg_{k}": v for k, v in avg_metrics.items()},
            # 添加当前任务的指标
            # **{f"curr_{k}": v for k, v in current_task_metrics.items()}
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
                self.recall_matrix.append([]); self.cost_matrix.append([]); self.f1_binary_matrix.append([]); self.precision_matrix.append([])
                continue
            
            self.task_valid_indices_map[task_id] = task_valid_idx.cpu().numpy()
            
            # 1. 获取当前任务的训练标签
            y_curr = self.dataset.y[task_train_idx] # 只看当前任务的训练集

            # 2. 计算比例
            num_pos = (y_curr == 1).sum().item()
            num_neg = (y_curr == 0).sum().item()

            if num_pos > 0:
                raw_ratio = num_neg / num_pos
            else:
                raw_ratio = 1.0

            # 3. 截断 (Clip) - 防止权重过大
            # 这里的 5.0 是给基线实验设的安全上限，您可以根据需要调整
            clipped_weight = min(raw_ratio, 5.0) 
            clipped_weight = max(clipped_weight, 1.0)

            print(f"🔧 Task {task_id+1} 动态权重: {raw_ratio:.2f} -> 截断为 {clipped_weight:.2f}")

            # 4. 重新定义 Loss 函数 (覆盖 __init__ 里的定义)
            # 注意：这里要用 .double() 还是 .float() 取决于您的模型精度，一般 float 即可
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([clipped_weight]).to(self.device)
            )

            # [SNAPSHOT 构建]
            current_max_step = time_steps[-1]
            valid_node_mask = self.dataset.timesteps <= current_max_step
            row, col = self.dataset.edge_index
            edge_mask = valid_node_mask[row] & valid_node_mask[col]
            snapshot_data = copy.copy(self.dataset)
            snapshot_data.edge_index = self.dataset.edge_index[:, edge_mask]

            if self.config.train.model == 'hogrl':
                print(f">>> [HOGRL] Pre-computing high-order graphs for Task {task_id+1} (Snapshot only)...")
                order = self.config.model.get('num_orders', 3)
                # 关键：传入的是 snapshot_data.edge_index
                snapshot_data.adjs = self._precompute_high_order_graphs(
                    snapshot_data.edge_index, self.dataset.num_nodes, order=order
                )

            is_ewc_mode = self.ewc_lambda > 0.0
            is_lwf_mode = self.lwf_alpha > 0.0
            is_replay_mode = self.replay_buffer.buffer_size_per_class > 0
            
            # [ConsisGAD] LGA Pre-training
            if self.config.train.model == 'consisgad':
                print(">>> [ConsisGAD] Pre-training LGA Module (on Snapshot)...")
                for lga_ep in range(20): 
                    ld, lg = self._train_lga_step(snapshot_data)
                    if lga_ep % 10 == 0: print(f"    LGA Ep {lga_ep}: D={ld:.3f} G={lg:.3f}")

            # --- Epoch Loop ---
            for epoch in range(1, epochs_per_task + 1):
                global_step += 1
                self.model.train()
                self.optimizer.zero_grad()
                
                replay_idx = self.replay_buffer.get_buffer_indices().to(self.config.train.device)
                current_train_idx = torch.cat([task_train_idx, replay_idx])
                
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

                # out_res = self.model(snapshot_data)
                # if isinstance(out_res, tuple):
                #     if len(out_res) == 2: outputs, proj_features = out_res 
                #     elif len(out_res) == 3: outputs, z_all, alpha_all = out_res 
                #     else: outputs = out_res[0]
                # else:
                #     outputs = out_res

                # 先初始化所有可能的辅助变量为 None
                # 这样即使模型没有返回它们，后续的 if 判断也不会报错
                z_all = None
                alpha_all = None
                proj_features = None

                # 执行模型前向传播
                out_res = self.model(snapshot_data)
                
                # 安全解包 (根据返回值的数量自动赋值)
                if isinstance(out_res, tuple):
                    if len(out_res) == 3:
                        # BSL 模型通常返回 3 个值: (logits, embeddings, attention_weights)
                        outputs, z_all, alpha_all = out_res
                    elif len(out_res) == 2:
                        # GradGNN 或其他模型可能返回 2 个值: (logits, projected_features)
                        outputs, proj_features = out_res 
                    else:
                        #以此类推，或者只取第一个
                        outputs = out_res[0]
                else:
                    # 普通模型只返回 logits
                    outputs = out_res
                
                # Reshape logits 确保维度正确
                outputs = outputs.reshape((self.dataset.x.shape[0]))
                
                outputs = outputs.reshape((self.dataset.x.shape[0]))
                task_y = self.dataset.y[current_train_idx].float().reshape(-1, 1)
                
                if self.config.train.model == 'gat_cobo':
                    sw = torch.ones_like(task_y); sw[task_y==1] = 10.0
                    task_loss = F.binary_cross_entropy_with_logits(
                        outputs[current_train_idx].double().reshape(-1, 1), task_y, weight=sw, pos_weight=self.criterion.pos_weight)
                else:
                    task_loss = self.criterion(outputs[current_train_idx].reshape(-1, 1), task_y)
                
                grad_loss, bsl_loss, consis_loss, cl_loss = 0.0, 0.0, 0.0, 0.0
                
                # [ConsisGAD - NAIVE BASELINE MODE]
                if self.config.train.model == 'consisgad' and consis_view_outputs is not None:
                    all_idx = torch.arange(self.dataset.num_nodes, device=self.config.train.device)
                    is_not_train = ~torch.isin(all_idx, current_train_idx)
                    is_visible = valid_node_mask
                    # Naive: 限制只能看当前任务时间段
                    task_start_t, task_end_t = time_steps[0], time_steps[-1]
                    is_current_task_nodes = (self.dataset.timesteps >= task_start_t) & \
                                            (self.dataset.timesteps <= task_end_t)
                    
                    unlabeled_mask = is_not_train & is_visible & is_current_task_nodes

                    if unlabeled_mask.sum() > 0:
                        p_orig = torch.sigmoid(outputs[unlabeled_mask])
                        p_aug = torch.sigmoid(consis_view_outputs.reshape(-1)[unlabeled_mask])
                        consis_loss = F.mse_loss(p_orig, p_aug)

                if self.config.train.model == 'grad' and proj_features is not None:
                    grad_loss = self._sup_contrastive_loss(proj_features[current_train_idx], self.dataset.y[current_train_idx])
                
                if self.config.train.model == 'bsl' and z_all is not None:
                    alpha_train = alpha_all[current_train_idx]
                    y_train_long = self.dataset.y[current_train_idx].long()
                    l_attn = torch.mean(
                        (1 - y_train_long.float()) * F.relu(alpha_train[:, 1] - alpha_train[:, 2] + 0.2) + 
                        (y_train_long.float()) * F.relu(alpha_train[:, 2] - alpha_train[:, 1] + 0.2)
                    )
                    bsl_loss = 0.4 * l_attn 
                    
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
                if self.config.train.model == 'grad': total_loss += 0.5 * grad_loss
                if self.config.train.model == 'bsl': total_loss += bsl_loss
                if self.config.train.model == 'consisgad': total_loss += 1.0 * consis_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step(total_loss)

                # [RL Auto-Tuning]
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
                    elif metrics['precision'] < 0.3: nw = max(cw * 0.8, 1.0) # 放宽下限
                    
                    if nw != cw: 
                         self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([nw]).to(self.config.train.device).double())

            # --- Task End ---
            if is_ewc_mode: self._update_ewc_metrics(task_train_idx, snapshot_data)
            if is_lwf_mode:
                self.old_model = copy.deepcopy(self.model)
                self.old_model.to(self.config.train.device)
            if is_replay_mode:
                task_train_labels = self.dataset.y[task_train_idx].cpu().numpy()
                self.replay_buffer.add_exemplars(task_train_idx.cpu().numpy(), task_train_labels)

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
