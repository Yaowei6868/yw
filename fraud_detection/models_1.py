import torch
import torch.nn as nn
import torch.nn.functional as F  
from torch.nn import GRUCell
import math
import random
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv, GATv2Conv, GraphConv, MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, dense_to_sparse

class LGAGenerator(nn.Module):
    """
    [ConsisGAD] Learnable Graph Augmentation Generator
    Input: Graph (x, edge_index)
    Output: Augmented Adjacency Matrix (Probabilities or Sampled)
    """
    def __init__(self, input_dim, hidden_dim, device):
        super(LGAGenerator, self).__init__()
        self.device = device
        # 简单的 GCN 编码器
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        # 1. 编码节点特征
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index) # [N, hidden_dim]
        
        # 2. 计算成对相似度作为边概率 P_ij = sigmoid(h_i^T * h_j)
        # 注意：对于大图，全图 N*N 计算会爆显存。
        # 优化策略：只针对现有边和部分随机负采样边计算，或者使用稀疏矩阵操作。
        # 这里为了演示逻辑，假设图较小使用 dense；大图需改为稀疏实现。
        prob_adj = torch.sigmoid(torch.matmul(h, h.t()))
        return prob_adj

    def sample_adj(self, prob_adj, temperature=1.0):
        """
        Gumbel-Softmax 采样 (Differentiable)
        """
        # Gumbel Noise
        eps = 1e-20
        u = torch.rand_like(prob_adj)
        g = -torch.log(-torch.log(u + eps) + eps)
        
        # Reparameterization trick
        # P(edge=1) = prob_adj
        # Logits for class 1: log(prob_adj)
        # Logits for class 0: log(1 - prob_adj)
        # 这里的实现简化为二元 Gumbel Softmax 近似
        # soft_sample = sigmoid((log(p) - log(1-p) + g) / temp)
        
        logits = torch.log(prob_adj + eps) - torch.log(1 - prob_adj + eps)
        adj_sampled = torch.sigmoid((logits + g) / temperature)
        return adj_sampled

class LGADiscriminator(nn.Module):
    """
    [ConsisGAD] Graph Discriminator
    Input: Graph (x, adj)
    Output: Realness Score (Scalar)
    """
    def __init__(self, input_dim, hidden_dim):
        super(LGADiscriminator, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight=None):
        # 支持加权边 (生成的软邻接矩阵)
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        
        # Graph-level pooling (Readout)
        # batch 向量假设全 0 (单图模式)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h_g = global_mean_pool(h, batch) # [1, hidden_dim]
        
        score = torch.sigmoid(self.lin(h_g))
        return score

class ConsisGAD(nn.Module):
    """
    ConsisGAD 主模型
    包含 Classifier, Generator, Discriminator
    """
    def __init__(self, config):
        super(ConsisGAD, self).__init__()
        self.config = config
        self.device = config.train.device
        
        # 1. GAD 分类器 (Backbone, e.g., GCN/GAT)
        # 这里复用 GCN 或 GAT 结构，可以直接调用现有的
        from .models import GCN # 假设已有
        self.classifier = GCN(config) # 或者定义一个新的简单 GCN
        
        # 2. LGA 组件
        self.generator = LGAGenerator(config.input_dim, 64, self.device)
        self.discriminator = LGADiscriminator(config.input_dim, 64)

    def forward(self, data):
        # 默认只做分类
        return self.classifier(data)

class PMPLayer(MessagePassing):
    """
    Partitioning Message Passing Layer (ICLR 2024) [Fixed]
    修复了 node_type 维度报错问题
    """
    def __init__(self, in_dim, out_dim):
        super(PMPLayer, self).__init__(aggr='add') # 使用 sum 聚合
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 1. 权重矩阵 (Shared parameters)
        self.W_fr = nn.Linear(in_dim, out_dim, bias=False)
        self.W_be = nn.Linear(in_dim, out_dim, bias=False)
        
        # 2. Alpha 生成器 (用于混合 Unlabeled Neighbors)
        # alpha = Sigmoid(MLP(x))
        self.alpha_mlp = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid()
        )
        
        # Self-loop 更新
        self.W_self = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, y, train_mask):
        """
        x: [N, in_dim]
        y: [N]
        train_mask: [N] (bool)
        """
        # 1. 计算 Alpha [N, 1]
        alpha = self.alpha_mlp(x) 
        
        # 2. 准备 node_type
        # 0=Benign, 1=Fraud, 2=Unlabeled
        # [关键修复] node_type 必须是 2D 张量 [N, 1]，否则 propagate 会报错
        node_type = torch.full_like(y, 2).unsqueeze(-1) # [N, 1]
        
        # 填充已知标签 (注意 y 也要 reshape 才能赋值)
        if train_mask is not None:
            # 确保 y 是 tensor
            if not torch.is_tensor(y):
                y = torch.tensor(y, device=x.device)
            # 赋值
            node_type[train_mask] = y[train_mask].unsqueeze(-1).to(node_type.dtype)
        
        # 3. 消息传递
        # x: [N, D], node_type: [N, 1], alpha: [N, 1]
        return self.propagate(edge_index, x=x, node_type=node_type, alpha=alpha)

    def message(self, x_j, node_type_j, alpha_i):
        """
        x_j: [E, in_dim]
        node_type_j: [E, 1] (来自 source node 的类型)
        alpha_i: [E, 1] (来自 target node 的 alpha)
        """
        # 计算三种基础消息
        msg_fr = self.W_fr(x_j) # [E, out_dim]
        msg_be = self.W_be(x_j) # [E, out_dim]
        
        # 计算 Unlabeled 的混合消息
        # alpha_i 是 [E, 1], msg 是 [E, out_dim], 广播乘法
        msg_un = alpha_i * msg_fr + (1 - alpha_i) * msg_be
        
        # [关键修复] 选择消息
        # node_type_j 已经是 [E, 1] 了，直接用 eq 比较，不用再 unsqueeze
        mask_fr = (node_type_j == 1) # [E, 1]
        mask_be = (node_type_j == 0) # [E, 1]
        mask_un = (node_type_j == 2) # [E, 1]
        
        final_msg = torch.zeros_like(msg_fr)
        
        # 使用 where 进行并行选择 (Broadcasting [E, 1] -> [E, out_dim])
        final_msg = torch.where(mask_fr, msg_fr, final_msg)
        final_msg = torch.where(mask_be, msg_be, final_msg)
        final_msg = torch.where(mask_un, msg_un, final_msg)
        
        return final_msg

    def update(self, aggr_out, x):
        # 加上 Self-loop
        return aggr_out + self.W_self(x)

class PMPModel(nn.Module):
    def __init__(self, config):
        super(PMPModel, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        
        # PMP 建议单层或双层
        self.conv1 = PMPLayer(self.input_dim, self.hidden_dim)
        self.conv2 = PMPLayer(self.hidden_dim, self.hidden_dim)
        
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 获取 Mask (由 Trainer 注入到 data.pmp_mask)
        if hasattr(data, 'pmp_mask'):
            mask = data.pmp_mask
        else:
            # Fallback: 如果没有 mask，全设为 False (全当 Unlabeled 处理)
            mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)

        # 第一层
        h = self.conv1(x, edge_index, data.y, mask)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 第二层
        h = self.conv2(h, edge_index, data.y, mask)
        h = F.relu(h)
        
        return self.classifier(h)

class BSL(nn.Module):
    """
    AAAI 2024: Barely Supervised Learning for Graph-Based Fraud Detection
    """
    def __init__(self, config):
        super(BSL, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim # 注意：hidden_dim 必须能被 3 整除
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        self.num_heads = config.get('num_heads', 4)
        
        # 确保隐藏层维度可以被3整除 (对应 NA, AA, NN 三个子空间)
        assert self.hidden_dim % 3 == 0, "Hidden dim must be divisible by 3 for BSL disentanglement."
        self.sub_dim = self.hidden_dim // 3

        # 1. 嵌入模块 (GNN Encoder) - 使用 GATv2
        self.gnn_encoder = GATv2Conv(self.input_dim, self.hidden_dim, heads=self.num_heads, concat=False)
        
        # 2. 解耦模块 (Disentanglement)
        # 边缘分类器 (用于 Edge Loss): 输入两个节点的子特征，判断边类型(NA/AA/NN)
        # 这里简化为 3 个独立的线性层，分别处理三个子空间
        self.edge_classifier_na = nn.Linear(self.sub_dim * 2, 1) # NA 子空间判断 NA 边
        self.edge_classifier_aa = nn.Linear(self.sub_dim * 2, 1) # AA 子空间判断 AA 边
        self.edge_classifier_nn = nn.Linear(self.sub_dim * 2, 1) # NN 子空间判断 NN 边

        # 注意力机制 (用于合并三个子空间)
        # Calculate weights for NA, AA, NN parts
        self.att_vec = nn.Parameter(torch.randn(3, self.sub_dim)) 
        self.att_bias = nn.Parameter(torch.zeros(3, 1))
        
        # 3. 最终分类器
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def get_sub_features(self, z):
        """将特征 z 切分为 [NA, AA, NN]"""
        return torch.chunk(z, 3, dim=1)

    def forward(self, data, return_parts=False):
        x, edge_index = data.x, data.edge_index
        
        # 1. GNN 编码
        z = self.gnn_encoder(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        
        # 2. 解耦与注意力聚合
        # z: [N, 3 * sub_dim]
        z_na, z_aa, z_nn = self.get_sub_features(z)
        
        # 计算 Attention Weights: alpha
        # 简单实现：W * z^T + b -> softmax
        # stack parts: [N, 3, sub_dim]
        z_stack = torch.stack([z_na, z_aa, z_nn], dim=1) 
        
        # [N, 3, sub_dim] * [3, sub_dim] (element-wise then sum?) 
        # 论文 Eq(7): w_i^k = q * sigma(W^k * z_i^k + b^k). 这里简化为线性变换.
        # 我们用一个简单的 Linear layer 模拟每个 subspace 的打分
        
        # 为了简单，我们对每个 subspace 计算一个分数
        # [N, 1]
        score_na = F.leaky_relu((z_na * self.att_vec[0]).sum(dim=1, keepdim=True) + self.att_bias[0])
        score_aa = F.leaky_relu((z_aa * self.att_vec[1]).sum(dim=1, keepdim=True) + self.att_bias[1])
        score_nn = F.leaky_relu((z_nn * self.att_vec[2]).sum(dim=1, keepdim=True) + self.att_bias[2])
        
        scores = torch.cat([score_na, score_aa, score_nn], dim=1) # [N, 3]
        alpha = F.softmax(scores, dim=1) # [N, 3]
        
        # 3. 加权聚合
        # z_final = alpha_na * z_na + alpha_aa * z_aa + alpha_nn * z_nn
        # 注意：论文 Eq(11) 是拼接还是加权求和？
        # 论文 Eq(11) 是 sum(alpha^k * z^k). 这意味着输出维度变为 sub_dim?
        # 但通常分类器需要足够的信息。如果 sub_dim 太小可能影响性能。
        # 我们这里假设是加权拼接 (Weighted Concatenation) 或者 保持 hidden_dim 维度
        # 论文图示 implying weighted aggregation based on disentangled parts.
        # 为了保持维度一致性方便 classifier，我们采用 element-wise weighting 然后拼接回去，或者直接加权
        # 让我们严格遵循 Eq 11: sum. 这样 output 是 sub_dim.
        # 为了复现效果，我们稍微修改：如果不降维，不仅利用 alpha 加权，保留拼接结构
        # z_final: [N, hidden_dim] = [alpha_0*z_na, alpha_1*z_aa, alpha_2*z_nn]
        z_weighted = torch.cat([
            z_na * alpha[:, 0:1],
            z_aa * alpha[:, 1:2],
            z_nn * alpha[:, 2:3]
        ], dim=1)
        
        out = self.classifier(z_weighted)
        
        if return_parts:
            # 返回 logits, 原始特征 z, 以及注意力权重 alpha (用于 Loss 计算)
            return out, z, alpha
        return out

class GradGNN(nn.Module):
    """
    Grad (WWW 2025) Simplified Version
    核心组件：
    1. GATv2 Encoder: 提取节点特征
    2. Projection Head: 将特征映射到对比学习空间
    """
    def __init__(self, config):
        super(GradGNN, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        self.num_heads = config.get('num_heads', 4)

        # 1. 骨干网络 (GATv2)
        self.conv1 = GATv2Conv(self.input_dim, self.hidden_dim, heads=self.num_heads)
        self.conv2 = GATv2Conv(self.num_heads * self.hidden_dim, self.hidden_dim, heads=self.num_heads, concat=False)
        
        # 2. 投影头 (用于对比学习)
        # 将特征映射到一个更适合计算相似度的空间
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim) # 保持维度一致方便计算
        )
        
        # 3. 分类器
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # GATv2 编码
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        # 注意：这里输出的是 embedding (h)
        h = F.relu(x)
        
        # 计算投影特征 (z) 用于对比 Loss
        z = self.projection_head(h)
        z = F.normalize(z, dim=1) # 归一化，这对对比学习很重要
        
        # 分类 logits
        out = self.classifier(h)
        
        # 返回: (分类结果, 投影特征, 节点嵌入)
        # Trainer 需要 z 来计算 Contrastive Loss
        return out, z

class DQNAgent(nn.Module):
    """
    一个简单的 DQN 智能体，用于动态调整训练超参数 (如 pos_weight)。
    """
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
        """Epsilon-greedy 策略选择动作"""
        if random.random() < epsilon:
            return random.randint(0, self.fc3.out_features - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
                q_values = self.forward(state_t)
                return torch.argmax(q_values).item()

class CGNN(nn.Module):
    """
    Context-aware Graph Neural Network (AAAI 2025) - Simplified for Reproduction
    核心思想：
    1. 维护类别原型 (Prototypes)
    2. 计算节点及其邻居与原型的相似度 (Semantic Context)
    3. 将语义信息注入特征，使用 GAT 进行去噪聚合
    """
    def __init__(self, config):
        super(CGNN, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        self.num_heads = config.get('num_heads', 4)
        self.num_prototypes = 2 # 欺诈检测通常是二分类 (0:正常, 1:欺诈)

        # 1. 特征变换
        self.lin_in = nn.Linear(self.input_dim, self.hidden_dim)
        
        # 2. 类别原型 (Learnable Class Prototypes)
        # 形状: [2, hidden_dim] -> 代表 "正常" 和 "欺诈" 的理想特征中心
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.hidden_dim))
        nn.init.xavier_uniform_(self.prototypes)

        # 3. 语义增强后的 GAT 层
        # 输入维度 = hidden_dim + num_prototypes (因为我们把相似度分数拼接到特征上了)
        self.gat = GATv2Conv(
            self.hidden_dim + self.num_prototypes, 
            self.hidden_dim, 
            heads=self.num_heads, 
            concat=False # 最后一层通常不拼接，或者拼接后投影
        )
        
        # 4. 分类器
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # --- 1. 特征投影 ---
        h = self.lin_in(x) # [N, hidden_dim]
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # --- 2. 类别语义分解 (CSD) ---
        # 计算每个节点与 2 个原型的相似度 (点积)
        # h: [N, D], prototypes: [2, D] -> scores: [N, 2]
        # 这里为了数值稳定性，可以做一下 normalize，或者直接用原始值
        h_norm = F.normalize(h, p=2, dim=1)
        p_norm = F.normalize(self.prototypes, p=2, dim=1)
        semantic_scores = torch.mm(h_norm, p_norm.t()) # [N, 2]
        
        # --- 3. 上下文语义增强 ---
        # 将 "该节点像好人的程度" 和 "像坏人的程度" 作为新特征拼上去
        # h_aug: [N, hidden_dim + 2]
        h_aug = torch.cat([h, semantic_scores], dim=1)
        
        # --- 4. 去噪注意力聚合 (DAM) ---
        # GATv2 会自动根据特征的相关性计算注意力权重
        # 因为特征里包含了语义分数，所以 Attention 会倾向于聚合语义一致的邻居
        out = self.gat(h_aug, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        
        # --- 5. 分类 ---
        return self.classifier(out)

class HOGRL(nn.Module):
    """
    High-Order Graph Representation Learning (Native PyTorch Version)
    """
    def __init__(self, config):
        super(HOGRL, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        self.num_orders = config.get('num_orders', 3)

        # 1. 多通道编码器
        self.encoders = nn.ModuleList([
            nn.Linear(self.input_dim, self.hidden_dim) 
            for _ in range(self.num_orders)
        ])
        
        # 2. 注意力融合层
        self.attn_vec = nn.Linear(self.hidden_dim, 1)
        
        # 3. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, data):
        x = data.x
        
        # 获取高阶图列表
        if hasattr(data, 'adjs'):
            adj_list = data.adjs
        else:
            # Fallback: 如果没有预计算，现场构造一个 A^1
            print("Warning: Using fallback adjacency matrix.")
            indices = data.edge_index
            values = torch.ones(indices.size(1)).to(x.device)
            adj = torch.sparse_coo_tensor(indices, values, (x.size(0), x.size(0))).to_sparse_csr()
            adj_list = [adj] * self.num_orders

        embeddings = []
        
        # 并行处理每个阶
        # 此时 adj_list 里的元素是 torch.sparse.csr_tensor
        for k in range(self.num_orders):
            # 1. 特征变换
            h = self.encoders[k](x)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # 2. 结构传播
            # 确保 adj_list 长度足够
            if k < len(adj_list):
                adj = adj_list[k]
                # [关键修改] 使用 PyTorch 原生矩阵乘法: Sparse @ Dense
                h = adj @ h 
            
            h = F.relu(h)
            embeddings.append(h)
            
        # 堆叠
        stack_emb = torch.stack(embeddings, dim=1)
        
        # 注意力融合
        attn_scores = torch.tanh(self.attn_vec(stack_emb))
        attn_weights = F.softmax(attn_scores, dim=1)
        final_emb = torch.sum(stack_emb * attn_weights, dim=1)
        
        # 分类
        return self.classifier(final_emb)

class GAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout

        self.conv1 = GATConv(self.input_dim, self.hidden_dim, heads=self.num_heads)
        self.relu1 = nn.ReLU()
        self.conv2 = GATConv(self.num_heads * self.hidden_dim, self.hidden_dim, heads=self.num_heads)
        self.relu2 = nn.ReLU()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.num_heads * self.hidden_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu1(x)
        x = self.conv2(x, edge_index)
        x = self.relu2(x)
        x = self.classifier(x)
        return x

class GraphSMOTE(nn.Module):
    def __init__(self, config):
        super(GraphSMOTE, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        
        # 扩充倍数 (例如 1.0 表示将少数类扩充 1 倍)
        self.up_scale = config.get('up_scale', 1.0) 

        # 1. 编码器 (Feature Extractor)
        self.encoder = GCNConv(self.input_dim, self.hidden_dim)
        
        # 2. 边生成器 (Edge Predictor)
        # 输入: 两个节点的 embedding 拼接 [2 * hidden_dim]
        # 输出: 边存在的概率 logits
        self.edge_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

        # 3. 分类器 (Classifier)
        self.classifier = GCNConv(self.hidden_dim, self.output_dim)

    def forward(self, data, mode='train'):
        x, edge_index = data.x, data.edge_index
        
        # --- Step 1: Encoding ---
        z = self.encoder(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        
        # 如果不是训练模式，直接分类，不进行 SMOTE
        if mode != 'train':
            out = self.classifier(z, edge_index)
            return out  # 仅返回 logits

        # --- Step 2: SMOTE ---
        # 仅对训练集中的少数类 (Label=1) 进行扩充
        # 注意：这里简化处理，假设传入的 data.y 是可访问的
        # 在 Trainer 中我们会传入 batch 或 mask，这里简单地基于全图 label 做索引
        
        y = data.y
        # 找到所有非法节点
        minority_idx = (y == 1).nonzero(as_tuple=True)[0]
        
        # 如果非法节点太少，无法插值，则跳过
        if len(minority_idx) < 2:
            out = self.classifier(z, edge_index)
            return out, torch.tensor(0.0).to(x.device)

        # 生成合成节点的特征
        z_new = self.smote(z, minority_idx)
        
        # --- Step 3: Edge Generation (关键步骤) ---
        # 1. 计算 Edge Predictor 的训练 Loss (让它学会判断边是否存在)
        ep_loss = self.get_edge_predictor_loss(z, edge_index)
        
        # 2. 为新节点建立连接
        # 为了效率，我们采用启发式策略：将新节点连接到部分真实的非法节点
        # 这样能让新节点的信息传播出去
        
        num_real = z.size(0)
        num_new = z_new.size(0)
        
        # 将新节点拼接到原节点后面
        z_all = torch.cat([z, z_new], dim=0)
        
        # 构建新边: 让每个新节点随机连接到一个真实的非法节点 (模拟同质性)
        # (在完整版 GraphSMOTE 中会用 edge_predictor 筛选，这里为了速度做简化)
        source_nodes = torch.arange(num_real, num_real + num_new, device=z.device)
        target_nodes = minority_idx[torch.randint(0, len(minority_idx), (num_new,), device=z.device)]
        
        new_edges = torch.stack([source_nodes, target_nodes], dim=0)
        # 添加双向边
        edge_index_all = torch.cat([edge_index, new_edges, new_edges.flip(0)], dim=1)
        
        # --- Step 4: Classification ---
        # 在扩充后的图上进行分类
        out = self.classifier(z_all, edge_index_all)
        
        # 返回: 
        # 1. 扩充后的 logits (Trainer 需要截取前 num_real 个用于计算真实节点的 loss)
        # 2. 边生成器的辅助 loss
        return out, ep_loss

    def smote(self, z, minority_idx):
        """在 Embedding 空间插值生成新节点"""
        num_minority = len(minority_idx)
        num_synthetic = int(num_minority * self.up_scale)
        
        # 随机选择种子节点 (Seeds)
        idx_seeds = minority_idx[torch.randint(0, num_minority, (num_synthetic,), device=z.device)]
        
        # 随机选择邻居节点 (Neighbors) - 简化版 KNN
        idx_neighbors = minority_idx[torch.randint(0, num_minority, (num_synthetic,), device=z.device)]
        
        # 插值系数
        alpha = torch.rand(num_synthetic, 1, device=z.device)
        
        z_seeds = z[idx_seeds]
        z_neighbors = z[idx_neighbors]
        
        # 插值公式: new = seed + alpha * (neighbor - seed)
        z_new = z_seeds + alpha * (z_neighbors - z_seeds)
        return z_new

    def get_edge_predictor_loss(self, z, edge_index):
        """训练边生成器区分真边和假边"""
        # 采样正样本 (存在的边)
        num_edges = edge_index.size(1)
        perm = torch.randperm(num_edges)[:1024] # 随机采 1024 条
        edge_pos = edge_index[:, perm]
        
        # 采样负样本 (随机两个点)
        edge_neg = torch.randint(0, z.size(0), (2, 1024), device=z.device)
        
        # 计算分数
        pos_scores = self.edge_predictor(torch.cat([z[edge_pos[0]], z[edge_pos[1]]], dim=1))
        neg_scores = self.edge_predictor(torch.cat([z[edge_neg[0]], z[edge_neg[1]]], dim=1))
        
        # Loss: 正样本标签1, 负样本标签0
        loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
               F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        return loss

class DOMINANT(nn.Module):
    def __init__(self, config):
        super(DOMINANT, self).__init__()
        # 接收 config 对象，与其他模型保持一致
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout

        # 1. 编码器 (Encoder): 类似于标准的 GCN
        self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)

        # 2. 属性解码器 (Attribute Decoder): 尝试从隐藏层还原原始特征
        self.attr_decoder_1 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.attr_decoder_2 = GCNConv(self.hidden_dim, self.input_dim)

        # 结构解码器 (Structure Decoder): 简单的点积，不需要额外参数
        # A_hat = Z * Z.T

    def forward(self, x, edge_index):
        # --- Encoding ---
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        z = self.conv2(x, edge_index)
        z = F.relu(z)
        # z 是学习到的节点嵌入 (Embeddings)

        # --- Decoding Attributes (特征重构) ---
        x_hat = self.attr_decoder_1(z, edge_index)
        x_hat = F.relu(x_hat)
        x_hat = F.dropout(x_hat, self.dropout, training=self.training)
        x_hat = self.attr_decoder_2(x_hat, edge_index)
        
        # 返回 重构特征 和 嵌入向量
        return x_hat, z

class GATv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout

        self.conv1 = GATv2Conv(self.input_dim, self.hidden_dim, heads=self.num_heads)
        self.conv2 = GATv2Conv(self.num_heads * self.hidden_dim, self.hidden_dim, heads=self.num_heads)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.num_heads * self.hidden_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout

        self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim) 
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        out = self.classifier(x)
        return out

class GIN(nn.Module):
    def __init__(self, config):
        super(GIN, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )
        )
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        ))
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x

class GraphSAGE(nn.Module):
    def __init__(self, config):
        super(GraphSAGE, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout

        self.conv1 = SAGEConv(self.input_dim, self.hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(self.hidden_dim, self.hidden_dim, aggr='mean')
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.classifier(x)
        return out

class STAGNN(nn.Module):
    def __init__(self, config):
        super(STAGNN, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.num_heads = config.get('num_heads', 4)
        self.dropout = config.dropout
        
        self.feature_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.time_embedding = nn.Embedding(100, self.hidden_dim)
        self.conv1 = GATConv(self.hidden_dim, self.hidden_dim, heads=self.num_heads, concat=True)
        self.classifier = nn.Linear(self.hidden_dim * self.num_heads, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.feature_proj(x)
        x = F.relu(x)
        
        if hasattr(data, 'timesteps') and data.timesteps is not None:
            t_emb = self.time_embedding(data.timesteps)
            x = x + t_emb
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.classifier(x)
        return out

class EvolveGCN(nn.Module):
    def __init__(self, config):
        super(EvolveGCN, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout

        self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.register_buffer("hidden_state", torch.zeros(1, self.hidden_dim)) 

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        graph_context = x.mean(dim=0, keepdim=True)
        if self.training:
             new_h = self.rnn(graph_context, self.hidden_state)
             self.hidden_state = new_h.detach()
        else:
             new_h = self.rnn(graph_context, self.hidden_state)
        x = x + new_h
        
        out = self.classifier(x)
        return out

class TimeEncode(nn.Module):
    def __init__(self, out_channels):
        super(TimeEncode, self).__init__()
        self.out_channels = out_channels
        self.w = nn.Linear(1, out_channels // 2)

    def forward(self, t):
        t = t.view(-1, 1).double() 
        out = self.w(t) 
        out = torch.cat([torch.cos(out), torch.sin(out)], dim=-1)
        return out

class TGN(nn.Module):
    def __init__(self, config):
        super(TGN, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.num_heads = config.get('num_heads', 2)
        self.dropout = config.dropout
        
        self.time_encoder = TimeEncode(self.hidden_dim)
        self.gat1 = GATConv(self.input_dim + self.hidden_dim, self.hidden_dim, heads=self.num_heads, concat=True)
        self.gat2 = GATConv(self.hidden_dim * self.num_heads, self.hidden_dim, heads=1, concat=False)
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if hasattr(data, 'timesteps') and data.timesteps is not None:
            t_emb = self.time_encoder(data.timesteps)
        else:
            t_emb = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        x = torch.cat([x, t_emb], dim=-1)
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.classifier(x)
        return out

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x = data.x
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.fc2(x)
        return out

