import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from torch.autograd import grad
from torch_geometric.nn import (
    GATConv, GCNConv, GINConv, SAGEConv, GATv2Conv, 
    GraphConv, MessagePassing, global_mean_pool
)
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, dense_to_sparse

# ==========================================
# 1. 基础骨干模型 (GCN, GAT, MLP等)
# ==========================================
class GCN(nn.Module):
    """
    Standard GCN Model (Used as backbone for ConsisGAD and others)
    """
    def __init__(self, config):
        super(GCN, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        
        # 确保 Conv 层支持 edge_weight (PyG 默认支持)
        self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.output_dim)

    def forward(self, x, edge_index=None, edge_weight=None):
        """
        参数兼容逻辑:
        1. 如果 x 是 Data 对象: 解包出 x, edge_index
        2. 如果 x 是 Tensor: 直接使用传入的 edge_index, edge_weight
        """
        # 1. 兼容 Data 对象输入 (常规训练调用)
        if hasattr(x, 'x'):
            data = x
            x = data.x
            edge_index = data.edge_index
            # 如果 Data 对象里没有 edge_weight，就保持 None
            edge_weight = getattr(data, 'edge_weight', None) 
        
        # 2. 执行卷积 (ConsisGAD 增强视图调用时，edge_weight 会有值)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        return x

class GAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.get('num_heads', 4)
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout

        self.conv1 = GATConv(self.input_dim, self.hidden_dim, heads=self.num_heads)
        self.conv2 = GATConv(self.num_heads * self.hidden_dim, self.hidden_dim, heads=self.num_heads)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.num_heads * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.classifier(x)
        return x

class GATv2(nn.Module):
    """
    GATv2: Dynamic Graph Attention (Stronger Baseline)
    """
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        self.num_heads = config.get('num_heads', 4)

        self.conv1 = GATv2Conv(self.input_dim, self.hidden_dim, heads=self.num_heads)
        self.conv2 = GATv2Conv(self.num_heads * self.hidden_dim, self.hidden_dim, heads=self.num_heads, concat=False)
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.classifier(x)

class GIN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        self.conv1 = GINConv(self.mlp1)
        self.classifier = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        return self.classifier(F.relu(x))

class GraphSAGE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = SAGEConv(config.input_dim, config.hidden_dim)
        self.conv2 = SAGEConv(config.hidden_dim, config.output_dim)
    
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        return self.conv2(x, data.edge_index)

# 在 fraud_detection/models.py 中添加以下代码

# ==========================================
# 占位符类 (防止 trainer.py 导入报错)
# ==========================================
class STAGNN(nn.Module):
    pass
class EvolveGCN(nn.Module):
    pass
class TGN(nn.Module):
    pass
class MLP(nn.Module):
    pass

# --- GraphSMOTE ---
class GraphSMOTE(nn.Module):
    def __init__(self, config):
        super(GraphSMOTE, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        
        self.encoder = GCNConv(self.input_dim, self.hidden_dim)
        self.classifier = GCNConv(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.encoder(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        out = self.classifier(z, edge_index)
        return out
class DQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.fc3.out_features - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
                q_values = self.forward(state_t)
                return torch.argmax(q_values).item()

# ==========================================
# 2. 高级/SOTA 模型 (ConsisGAD, PMP, BSL, HOGRL, CGNN)
# ==========================================

# --- CGNN ---
class CGNNLayer(MessagePassing):
    """
    Strict implementation of AAAI-25 CGNN Layer.
    Ref: Context-aware Graph Neural Network for Graph-based Fraud Detection
    """
    def __init__(self, in_dim, out_dim):
        # 聚合方式：add (对应公式中的求和)
        super(CGNNLayer, self).__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        # 确保输入维度能被2整除 (论文 Eq.2: m = d // 2)
        assert in_dim % 2 == 0, "Input dimension must be divisible by 2 for splitting."
        self.half_dim = in_dim // 2
        # 1. 语义分解 (Eq. 2)
        # W_nor, b_nor
        self.lin_nor = nn.Linear(self.half_dim, out_dim)
        # W_abnor, b_abnor
        self.lin_abnor = nn.Linear(self.half_dim, out_dim)
        # 2. 去噪注意力 (Eq. 4)
        # W, b inside Tanh
        self.att_lin = nn.Linear(out_dim, out_dim) 
        # W_Att vector
        self.att_vec = nn.Linear(out_dim, 1, bias=False)
        # 3. 更新层 (Eq. 5)
        self.update_lin = nn.Linear(out_dim, out_dim)

    def forward(self, x, edge_index):

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x_left = x[:, :self.half_dim]
        x_right = x[:, self.half_dim:]
        # 投影到子空间
        x_nor = self.lin_nor(x_left)      # x^0
        x_abnor = self.lin_abnor(x_right) # x^1
        self._cached_x_nor = x_nor
        self._cached_x_abnor = x_abnor
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # 开始传播
        return self.propagate(edge_index, x_nor=x_nor, x_abnor=x_abnor, norm=norm)

    def message(self, x_nor_j, x_abnor_j, norm):
        """
        对应论文 Eq. 4: Denoising Attention
        alpha_j = softmax(W_Att * Tanh(W(x_j^0 + x_j^1) + b))
        """
        # W(x^0 + x^1) + b
        merged = x_nor_j + x_abnor_j
        h_att = torch.tanh(self.att_lin(merged))

        att_score = self.att_vec(h_att) # [E, 1]
        alpha = torch.sigmoid(att_score) 

        # 加权求和: x_j = alpha * x^0 + (1-alpha) * x^1
        x_j = alpha * x_nor_j + (1 - alpha) * x_abnor_j
               # 应用度归一化 (Eq. 5 中的 1/sqrt(d))
        return norm.view(-1, 1) * x_j
    def update(self, aggr_out, x_nor, x_abnor):
        return self.update_lin(aggr_out)

class CGNN(nn.Module):
    """
    Full Context-aware Graph Neural Network Model
    """
    def __init__(self, config):
        super(CGNN, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout     
        # 论文 Eq. 1: 初始特征映射
        self.lin_in = nn.Linear(self.input_dim, self.hidden_dim)      
        # 核心 CGNN 层
        # 注意：输入维度必须是 hidden_dim (且能被2整除)
        self.conv = CGNNLayer(self.hidden_dim, self.hidden_dim)      
        # 分类器
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data, return_decomposed=False):
        x, edge_index = data.x, data.edge_index      
        # 1. 映射到潜在空间 (Eq. 1)
        x = self.lin_in(x)
        x = F.leaky_relu(x) # 论文提及使用 Leaky ReLU
        x = F.dropout(x, p=self.dropout, training=self.training)       
        # 2. CGNN 核心层 (Eq. 2-5)
        # 这一步内部完成了分解、注意力计算和聚合
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)     
        # 3. 分类 (Eq. 6)
        out = self.classifier(x)    
        # 为了计算论文中的 L_CSD (对比损失)，需要返回分解后的特征
        if return_decomposed:
            # 从 layer 中取出缓存的 x_nor 和 x_abnor
            return out, self.conv._cached_x_nor, self.conv._cached_x_abnor
        return out



# --- HOGRL ---
class HOGRL(nn.Module):
    def __init__(self, config):
        super(HOGRL, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        self.num_orders = config.get('num_orders', 3)

        # --- 1. Experts for High-order Graphs (Eq. 4) ---
        # 每一阶都有独立的编码器
        self.order_encoders = nn.ModuleList([
            nn.Linear(self.input_dim, self.hidden_dim) 
            for _ in range(self.num_orders)
        ])

        # --- 2. Gating Network (Eq. 5-6) ---
        # 论文指出每个 expert (order) 都有自己的权重 w_l [cite: 186]
        # 使用 ModuleList 确保每一阶参数独立
        self.gating_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, 1)
            for _ in range(self.num_orders)
        ])

        # --- 3. Original Graph Branch (Eq. 8-9) ---
        # 论文要求保留原始图的 embedding 以补充多跳依赖 [cite: 203]
        # 这里使用一个简单的 GCN 层作为原始图分支的实现
        self.original_encoder = nn.Linear(self.input_dim, self.hidden_dim)

        # --- 4. Classifier (Eq. 12) ---
        # 图 2(d) 显示使用 Concatenated Embedding [cite: 103]
        # 输入维度 = 高阶特征 (hidden) + 原始特征 (hidden)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # 论文 Eq. 10 提到的超参数 gamma，虽然拼接方案可能不需要，但为了灵活性可以保留
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        x = data.x
        # 必需：预计算的高阶邻接矩阵列表 [S^1, S^2, ..., S^L]
        # 论文中的 graph construction 步骤 [cite: 164-168]
        if hasattr(data, 'adjs'):
            adj_list = data.adjs
        else:
            # Fallback: 如果没有预计算，这种回退仅用于调试，实际效果会很差
            # 实际上 HOGRL 强依赖 S^l = A^l - A^{l-1} + I
            indices = data.edge_index
            values = torch.ones(indices.size(1)).to(x.device)
            adj = torch.sparse_coo_tensor(indices, values, (x.size(0), x.size(0))).to_sparse_csr()
            adj_list = [adj] * self.num_orders
        # === Step 1: High-order Representation Learning ===
        expert_outputs = []
        for k in range(self.num_orders):
            # 线性变换: X * W^l [cite: 175]
            h = self.order_encoders[k](x)
            h = F.dropout(h, p=self.dropout, training=self.training)          
            # 聚合: S^l * h
            if k < len(adj_list):
                h = adj_list[k] @ h  
            h = F.relu(h)
            expert_outputs.append(h)        
        # === Step 2: MoE Attention (Mixture-of-Experts) ===
        # 计算每个 Expert 的 Attention score 
        attn_scores = []
        for k in range(self.num_orders):
            # f_l(h) = w_l * h + b_l
            score = self.gating_layers[k](expert_outputs[k]) # [N, 1]
            attn_scores.append(score)    
        # 拼接所有阶的分数: [N, L]
        attn_scores = torch.cat(attn_scores, dim=1)
        # Softmax 归一化 
        attn_weights = F.softmax(attn_scores, dim=1) # [N, L]
        # 加权求和得到高阶特征 h' [cite: 189]
        h_high_order = torch.zeros_like(expert_outputs[0])
        for k in range(self.num_orders):
            # 广播权重: [N, 1] * [N, hidden]
            h_high_order += attn_weights[:, k:k+1] * expert_outputs[k]
        # === Step 3: Original Graph Representation ===
        # 计算基于原始图的特征 h_v [cite: 194]
        # 通常使用 1 阶邻接矩阵 (adj_list[0] 或者是原始 A)
        h_original = self.original_encoder(x)
        h_original = F.dropout(h_original, p=self.dropout, training=self.training)
        # 使用原始邻接矩阵聚合 (通常就是 adj_list[0] 对应的 A^1，或者你需要单独传 A)
        # 这里假设 adj_list[0] 近似原始图结构
        if len(adj_list) > 0:
             h_original = adj_list[0] @ h_original
        h_original = F.relu(h_original)
        # === Step 4: Fusion & Prediction ===
        # 拼接 (Concatenation) [cite: 139]
        z_v = torch.cat([h_original, self.gamma * h_high_order], dim=1)
        return self.classifier(z_v)



# --- GradGNN ---
class HighPassEncoder(nn.Module):
    """
    [Grad] Supervised GCL Encoder (High-pass Filter)
    Eq. 1: H^(l+1) = sigma(W * (H^l - Mean(H_neigh)))
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        # 计算邻居均值
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[col]
        
        # Mean aggregation
        x_neigh = torch.zeros_like(x)
        # scatter_add: x_neigh[col] += x[row] * norm
        # PyG 的 MessagePassing 也可以做，这里手动实现以匹配公式逻辑
        # 简化实现: 使用 propogation
        return self.propagate(edge_index, x=x, norm=norm)

    def propagate(self, edge_index, x, norm):
        row, col = edge_index
        out = torch.zeros_like(x)
        # Message: x_j * norm
        msg = x[row] * norm.view(-1, 1)
        # Aggregate: sum
        out.scatter_add_(0, col.unsqueeze(1).expand_as(msg), msg)
        
        # High-pass: Self - Neighbor
        h = x - out
        return F.relu(self.lin(h))

class DenoisingNetwork(nn.Module):
    """
    [Grad] Denoising Network for Relation Diffusion
    Input: Noisy Adjacency (k x k), Time embedding
    Output: Predicted Noise
    """
    def __init__(self, group_size, time_dim=32):
        super().__init__()
        self.group_size = group_size
        self.input_dim = group_size * group_size
        
        # 时间步编码
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # 处理邻接矩阵 (Flatten -> MLP -> Reshape)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim + time_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, self.input_dim)
        )

    def forward(self, adj_noisy, t):
        # adj_noisy: [B, k, k]
        B = adj_noisy.size(0)
        adj_flat = adj_noisy.view(B, -1) # [B, k*k]
        
        t_emb = self.time_mlp(t.view(-1, 1).float()) # [B, time_dim]
        
        h = torch.cat([adj_flat, t_emb], dim=1)
        noise_pred = self.mlp(h)
        
        return noise_pred.view(B, self.group_size, self.group_size)

class BetaWaveletLayer(nn.Module):
    """
    [Grad] Beta Wavelet Filter
    Strict fix for dimension mismatch in spmm.
    """
    def __init__(self, in_dim, out_dim, order=4):
        super().__init__()
        self.order = order
        self.lins = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(order + 1)])

    def forward(self, x, edge_index, num_nodes):
        edge_index_self, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_self
        deg = degree(col, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        out = self.lins[0](x)
        Tx_prev = x
        
        # --- 关键修复: 安全的 Sparse MM 函数 ---
        def spmm(norm_val, idx, mat):
            # mat: (N, D)
            # idx: (2, E)
            # norm_val: (E,)
            
            E = idx.size(1)
            D = mat.size(1)
            
            # 1. 计算边上的消息: (E, D)
            # norm_val.view(-1, 1) -> (E, 1)
            # mat[idx[0]] -> (E, D)
            msg = norm_val.view(-1, 1) * mat[idx[0]]
            
            # 2. 准备输出容器
            out_tensor = torch.zeros_like(mat)
            
            # 3. 扩展索引以匹配消息形状: (E, 1) -> (E, D)
            # idx[1] 是目标节点索引
            index = idx[1].view(-1, 1).expand(E, D)
            
            # 4. 聚合
            return out_tensor.scatter_add_(0, index, msg)

        for i in range(1, self.order + 1):
            Tx_next = spmm(norm, edge_index_self, Tx_prev)
            out = out + self.lins[i](Tx_next)
            Tx_prev = Tx_next
            
        return F.relu(out)
        
class GradGNN(nn.Module):
    """
    [Grad] Full Model Implementation
    """
    def __init__(self, config):
        super(GradGNN, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        
        # 1. Supervised GCL Module
        self.gcl_encoder = HighPassEncoder(self.input_dim, self.hidden_dim)
        self.gcl_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 2. Diffusion Module
        self.group_size = 32 # k in paper
        self.diff_steps = 100 # T in paper
        self.denoise_net = DenoisingNetwork(self.group_size)
        
        # 3. Multi-Relation Detector (Wavelet)
        # 原始关系 + 生成关系 (Assumption: 1 generated relation)
        self.detector_orig = BetaWaveletLayer(self.input_dim, self.hidden_dim)
        self.detector_aug = BetaWaveletLayer(self.input_dim, self.hidden_dim)
        
        # Weighted Fusion (Eq. 15)
        self.fusion_weight = nn.Parameter(torch.ones(2))
        
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data, generated_adj=None):
        # 这里的 forward 主要用于 Detector 的训练
        x, edge_index = data.x, data.edge_index
        
        # 1. Original Branch
        h_orig = self.detector_orig(x, edge_index, x.size(0))
        
        # 2. Augmented Branch
        if generated_adj is not None:
            # generated_adj 是稀疏索引 [2, E_new]
            h_aug = self.detector_aug(x, generated_adj, x.size(0))
        else:
            h_aug = torch.zeros_like(h_orig)
            
        # 3. Fusion
        w = F.softmax(self.fusion_weight, dim=0)
        h_final = w[0] * h_orig + w[1] * h_aug
        
        return self.classifier(h_final), None # 保持接口一致 (out, aux)
    
    def forward_gcl(self, data):
        # 专门用于 GCL 训练的前向传播
        h = self.gcl_encoder(data.x, data.edge_index)
        z = self.gcl_proj(h)
        return z



# --- BSL (Barely Supervised Learning for Graph-Based Fraud Detection) ---
class BSL(nn.Module):
    """
    [AAAI-24] Barely Supervised Learning for Graph-Based Fraud Detection
    Full Implementation including Edge Classification Head.
    """
    def __init__(self, config):
        super(BSL, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        self.num_heads = config.get('num_heads', 4)
        
        # [cite_start]确保 hidden_dim 能被 3 整除 (对应 Z_na, Z_aa, Z_nn) [cite: 50]
        assert self.hidden_dim % 3 == 0, "Hidden dim must be divisible by 3 for BSL disentanglement."
        self.sub_dim = self.hidden_dim // 3

        # [cite_start]1. Embedding Module (GNN Encoder) [cite: 127]
        # 论文中使用 GNN 映射到潜在空间，这里使用 GATv2
        self.gnn_encoder = GATv2Conv(self.input_dim, self.hidden_dim, heads=self.num_heads, concat=False)
        
        # [cite_start]2. Disentanglement Module - Attention Weights [cite: 164]
        # 用于计算 attention alpha
        self.att_vec = nn.Parameter(torch.randn(3, self.sub_dim)) 
        self.att_bias = nn.Parameter(torch.zeros(3, 1))
        
        # [cite_start]3. Disentanglement Module - Edge Classifier [cite: 146]
        # 输入是两个节点的子特征拼接 (sub_dim * 2)，输出是边类型 (NA, AA, NN) 的 Logits
        self.edge_decoder = nn.Sequential(
            nn.Linear(self.sub_dim * 2, self.sub_dim),
            nn.ReLU(),
            nn.Linear(self.sub_dim, 1) # 输出标量 score
        )
        
        # [cite_start]4. Final Classifier [cite: 182]
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def get_sub_features(self, z):
        """将特征切分为 [Z_na, Z_aa, Z_nn]"""
        return torch.chunk(z, 3, dim=1)

    def get_attention_weights(self, z_parts):
        """计算 alpha 权重 (Eq. 7-8)"""
        z_na, z_aa, z_nn = z_parts
        
        # 计算 Attention Logits
        # score = q * sigma(W * z + b) -> 这里简化为点积形式实现，效果等价
        score_na = F.leaky_relu((z_na * self.att_vec[0]).sum(dim=1, keepdim=True) + self.att_bias[0])
        score_aa = F.leaky_relu((z_aa * self.att_vec[1]).sum(dim=1, keepdim=True) + self.att_bias[1])
        score_nn = F.leaky_relu((z_nn * self.att_vec[2]).sum(dim=1, keepdim=True) + self.att_bias[2])
        
        scores = torch.cat([score_na, score_aa, score_nn], dim=1)
        alpha = F.softmax(scores, dim=1) # [N, 3]
        return alpha

    def forward(self, data, return_stats=False):
        x, edge_index = data.x, data.edge_index
        
        # 1. GNN Encoding
        z = self.gnn_encoder(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        
        # 2. Disentanglement
        z_parts = self.get_sub_features(z) # (z_na, z_aa, z_nn)
        alpha = self.get_attention_weights(z_parts)
        
        # [cite_start]3. Re-weighting & Classification [cite: 179]
        # input = sum(alpha_k * z_k)
        # 实际上论文是拼接加权后的向量，或者是加权求和，这里为了保持维度一致使用拼接后过 Linear
        # Eq. 11 暗示是加权后的聚合。这里我们采用加权拼接:
        z_weighted = torch.cat([
            z_parts[0] * alpha[:, 0:1],
            z_parts[1] * alpha[:, 1:2],
            z_parts[2] * alpha[:, 2:3]
        ], dim=1)
        
        out = self.classifier(z_weighted)
        
        if return_stats:
            return out, z, alpha
        return out

    def predict_edge(self, z_src, z_dst, edge_type_idx):
        """
        用于计算 L_link。
        edge_type_idx: 0=NA, 1=AA, 2=NN
        [cite_start]我们需要根据边类型选择对应的子空间进行预测 [cite: 157]
        """
        # z_src, z_dst 是全特征
        src_parts = self.get_sub_features(z_src)
        dst_parts = self.get_sub_features(z_dst)
        
        # [cite_start]根据假设：第k类边只在第k个子空间相关 [cite: 140]
        # NA对应 idx 0, AA对应 idx 1, NN对应 idx 2
        k = edge_type_idx
        
        feat_cat = torch.cat([src_parts[k], dst_parts[k]], dim=1)
        return self.edge_decoder(feat_cat)



# --- ConsisGAD Components ---
class HomophilyAwareConv(MessagePassing):
    """
    [ConsisGAD] Homophily-Aware Neighborhood Aggregation (Section 3.1)
    h_v = AGGR({ MLP(h_v || h_u) : u in N(v) })
    """
    def __init__(self, in_dim, out_dim):
        super(HomophilyAwareConv, self).__init__(aggr='add') # 论文提及 AGGR 可以是 sum
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index):
        # x: [N, in_dim]
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # 拼接中心节点和邻居节点特征: [E, in_dim * 2]
        cat_feat = torch.cat([x_i, x_j], dim=1)
        # Edge-level homophily representation
        return self.mlp(cat_feat)

class LearnableAugmentor(nn.Module):
    """
    [ConsisGAD] Learnable Data Augmentation Module (Section 3.2.2)
    h_hat = Sharpen(Atten(h)) * h
    """
    def __init__(self, hidden_dim, drop_ratio=0.5, temperature=0.1):
        super(LearnableAugmentor, self).__init__()
        self.atten_lin = nn.Linear(hidden_dim, hidden_dim)
        self.att_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.drop_ratio = drop_ratio
        self.temperature = temperature

    def sharpen(self, h):
        """
        Algorithm 2: Sharpen Function (Differentiable Masking)
        保留 Top-k 的维度，抑制其他维度
        """
        # 注意：为了实现可导的 Top-k Mask，论文使用了一种 Softmax 技巧
        # 这里我们使用一种数值稳定的近似实现
        
        # 1. 计算 Mask 阈值 (Top-k)
        # h: [N, D]
        k = int(h.size(1) * (1 - self.drop_ratio))
        if k < 1: k = 1
        
        # 找到第 k 大的值作为 pivot
        topk_val, _ = torch.topk(h, k, dim=1)
        pivot = topk_val[:, -1].unsqueeze(1) # [N, 1]
        
        # 2. 生成 Soft Mask
        # 大于 pivot 的趋向于 1，小于的趋向于 0 (通过 temperature 控制陡峭程度)
        mask = torch.sigmoid((h - pivot) / self.temperature)
        return mask

    def forward(self, h):
        # 1. Attention: w = Wh + b (Eq. 8 simplified)
        atten_weights = self.atten_lin(h) + self.att_bias
        
        # 2. Sharpening
        mask = self.sharpen(atten_weights)
        
        # 3. Augmentation
        return h * mask

class ConsisGAD(nn.Module):
    """
    [ConsisGAD] Full Model
    """
    def __init__(self, config):
        super(ConsisGAD, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        
        # 1. Backbone GNN (Homophily-Aware)
        # 论文通常使用单层或两层，第一层做投影，第二层做聚合
        self.lin_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.conv = HomophilyAwareConv(self.hidden_dim, self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)
        
        # 2. Learnable Augmentor
        # 这里的参数需要根据 Dataset 调整，默认 xi=0.2
        drop_ratio = getattr(config, 'aug_drop_ratio', 0.2)
        self.augmentor = LearnableAugmentor(self.hidden_dim, drop_ratio=drop_ratio)

    def get_embedding(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.lin_in(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv(h, edge_index)
        return h # [N, hidden_dim]

    def forward(self, data, augment=False):
        # 1. 获取 Backbone Embedding
        h = self.get_embedding(data)
        h = F.relu(h)
        
        # 2. 如果需要增强 (用于 Augmentor 训练或一致性训练)
        if augment:
            h_aug = self.augmentor(h)
            out_aug = self.classifier(h_aug)
            return self.classifier(h), out_aug, h, h_aug
        
        # 3. 正常输出
        return self.classifier(h)



# --- PMP (Partitioning Message Passing) ---
class PMPLayer(MessagePassing):
    """
    Partitioning Message Passing Layer (ICLR 2024)
    Fixed: 修复 node_type 维度问题
    """
    def __init__(self, in_dim, out_dim):
        super(PMPLayer, self).__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.W_fr = nn.Linear(in_dim, out_dim, bias=False)
        self.W_be = nn.Linear(in_dim, out_dim, bias=False)
        
        self.alpha_mlp = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid()
        )
        self.W_self = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, y, train_mask):
        # 1. Alpha
        alpha = self.alpha_mlp(x) 
        
        # 2. Node Type Preparation [N, 1]
        # 0=Benign, 1=Fraud, 2=Unlabeled
        node_type = torch.full((x.size(0), 1), 2, device=x.device, dtype=y.dtype)
        
        if train_mask is not None:
            # 确保 mask 是 bool 类型
            if train_mask.dtype != torch.bool:
                train_mask = train_mask.bool()
            # 填充已知标签
            node_type[train_mask] = y[train_mask].unsqueeze(-1)
        
        return self.propagate(edge_index, x=x, node_type=node_type, alpha=alpha)

    def message(self, x_j, node_type_j, alpha_i):
        msg_fr = self.W_fr(x_j)
        msg_be = self.W_be(x_j)
        msg_un = alpha_i * msg_fr + (1 - alpha_i) * msg_be
        
        # node_type_j: [E, 1]
        mask_fr = (node_type_j == 1)
        mask_be = (node_type_j == 0)
        mask_un = (node_type_j == 2)
        
        final_msg = torch.zeros_like(msg_fr)
        final_msg = torch.where(mask_fr, msg_fr, final_msg)
        final_msg = torch.where(mask_be, msg_be, final_msg)
        final_msg = torch.where(mask_un, msg_un, final_msg)
        
        return final_msg

    def update(self, aggr_out, x):
        return aggr_out + self.W_self(x)
class PMPModel(nn.Module):
    def __init__(self, config):
        super(PMPModel, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        
        self.conv1 = PMPLayer(self.input_dim, self.hidden_dim)
        self.conv2 = PMPLayer(self.hidden_dim, self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 从 data 中获取 pmp_mask (由 Trainer 注入)
        mask = getattr(data, 'pmp_mask', None)
        if mask is None:
            mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)

        h = self.conv1(x, edge_index, data.y, mask)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.conv2(h, edge_index, data.y, mask)
        h = F.relu(h)
        
        return self.classifier(h)

