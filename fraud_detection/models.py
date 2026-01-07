import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
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

# 占位符类 (如果还需要用到)
class STAGNN(nn.Module): pass
class EvolveGCN(nn.Module): pass
class TGN(nn.Module): pass
class MLP(nn.Module): pass

# ==========================================
# 2. 高级/SOTA 模型 (ConsisGAD, PMP, BSL, HOGRL, CGNN)
# ==========================================

# --- ConsisGAD Components ---

class LGAGenerator(nn.Module):
    """
    [ConsisGAD] Learnable Graph Augmentation Generator
    Fixed: 移除 device 依赖，使用稀疏计算防止 OOM
    """
    def __init__(self, input_dim, hidden_dim):
        super(LGAGenerator, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        # 1. 编码
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        
        # 2. 仅计算现有边的保留概率 (Sparse Implementation)
        # 避免 N*N 稠密矩阵计算
        row, col = edge_index
        h_src = h[row]
        h_dst = h[col]
        # P_ij = sigmoid(h_i * h_j)
        score = (h_src * h_dst).sum(dim=-1)
        prob_edge = torch.sigmoid(score)
        
        return prob_edge

    def sample_adj(self, prob_edge, temperature=1.0):
        """Gumbel-Softmax Sampling for Edges"""
        eps = 1e-20
        u = torch.rand_like(prob_edge)
        g = -torch.log(-torch.log(u + eps) + eps)
        
        logits = torch.log(prob_edge + eps) - torch.log(1 - prob_edge + eps)
        # Soft sampling of edge weights
        edge_weight = torch.sigmoid((logits + g) / temperature)
        return edge_weight

class LGADiscriminator(nn.Module):
    """[ConsisGAD] Graph Discriminator"""
    def __init__(self, input_dim, hidden_dim):
        super(LGADiscriminator, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        # 全局池化得到图表示
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h_g = global_mean_pool(h, batch)
        score = torch.sigmoid(self.lin(h_g))
        return score

class ConsisGAD(nn.Module):
    """
    [ConsisGAD] Main Model
    Integrates GCN Classifier, LGA Generator, and Discriminator
    """
    def __init__(self, config):
        super(ConsisGAD, self).__init__()
        self.config = config
        
        # 1. Classifier (Backbone)
        self.classifier = GCN(config)
        
        # 2. LGA Components
        # 这里的 hidden_dim 可以设置小一点以节省显存
        lga_dim = getattr(config, 'lga_hidden_dim', 64)
        self.generator = LGAGenerator(config.input_dim, lga_dim)
        self.discriminator = LGADiscriminator(config.input_dim, lga_dim)

    def forward(self, data):
        # 默认前向传播只走分类器
        return self.classifier(data)

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

# --- BSL (Barely Supervised Learning) ---

class BSL(nn.Module):
    def __init__(self, config):
        super(BSL, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        self.num_heads = config.get('num_heads', 4)
        
        assert self.hidden_dim % 3 == 0, "Hidden dim must be divisible by 3."
        self.sub_dim = self.hidden_dim // 3

        self.gnn_encoder = GATv2Conv(self.input_dim, self.hidden_dim, heads=self.num_heads, concat=False)
        
        self.att_vec = nn.Parameter(torch.randn(3, self.sub_dim)) 
        self.att_bias = nn.Parameter(torch.zeros(3, 1))
        
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def get_sub_features(self, z):
        return torch.chunk(z, 3, dim=1)

    def forward(self, data, return_parts=False):
        x, edge_index = data.x, data.edge_index
        z = self.gnn_encoder(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        
        z_na, z_aa, z_nn = self.get_sub_features(z)
        
        score_na = F.leaky_relu((z_na * self.att_vec[0]).sum(dim=1, keepdim=True) + self.att_bias[0])
        score_aa = F.leaky_relu((z_aa * self.att_vec[1]).sum(dim=1, keepdim=True) + self.att_bias[1])
        score_nn = F.leaky_relu((z_nn * self.att_vec[2]).sum(dim=1, keepdim=True) + self.att_bias[2])
        
        scores = torch.cat([score_na, score_aa, score_nn], dim=1)
        alpha = F.softmax(scores, dim=1)
        
        z_weighted = torch.cat([
            z_na * alpha[:, 0:1],
            z_aa * alpha[:, 1:2],
            z_nn * alpha[:, 2:3]
        ], dim=1)
        
        out = self.classifier(z_weighted)
        
        if return_parts:
            return out, z, alpha
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

        self.encoders = nn.ModuleList([
            nn.Linear(self.input_dim, self.hidden_dim) 
            for _ in range(self.num_orders)
        ])
        self.attn_vec = nn.Linear(self.hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, data):
        x = data.x
        if hasattr(data, 'adjs'):
            adj_list = data.adjs
        else:
            indices = data.edge_index
            values = torch.ones(indices.size(1)).to(x.device)
            adj = torch.sparse_coo_tensor(indices, values, (x.size(0), x.size(0))).to_sparse_csr()
            adj_list = [adj] * self.num_orders

        embeddings = []
        for k in range(self.num_orders):
            h = self.encoders[k](x)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if k < len(adj_list):
                adj = adj_list[k]
                h = adj @ h 
            h = F.relu(h)
            embeddings.append(h)
            
        stack_emb = torch.stack(embeddings, dim=1)
        attn_scores = torch.tanh(self.attn_vec(stack_emb))
        attn_weights = F.softmax(attn_scores, dim=1)
        final_emb = torch.sum(stack_emb * attn_weights, dim=1)
        return self.classifier(final_emb)

# --- CGNN ---

class CGNN(nn.Module):
    def __init__(self, config):
        super(CGNN, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        self.num_heads = config.get('num_heads', 4)
        self.num_prototypes = 2

        self.lin_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.hidden_dim))
        nn.init.xavier_uniform_(self.prototypes)

        self.gat = GATv2Conv(self.hidden_dim + self.num_prototypes, self.hidden_dim, heads=self.num_heads, concat=False)
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.lin_in(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h_norm = F.normalize(h, p=2, dim=1)
        p_norm = F.normalize(self.prototypes, p=2, dim=1)
        semantic_scores = torch.mm(h_norm, p_norm.t())
        
        h_aug = torch.cat([h, semantic_scores], dim=1)
        out = self.gat(h_aug, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.classifier(out)

# --- GradGNN ---

class GradGNN(nn.Module):
    def __init__(self, config):
        super(GradGNN, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        self.num_heads = config.get('num_heads', 4)

        self.conv1 = GATv2Conv(self.input_dim, self.hidden_dim, heads=self.num_heads)
        self.conv2 = GATv2Conv(self.num_heads * self.hidden_dim, self.hidden_dim, heads=self.num_heads, concat=False)
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        h = self.conv2(x, edge_index)
        h = F.relu(h)
        
        z = self.projection_head(h)
        z = F.normalize(z, dim=1)
        
        out = self.classifier(h)
        return out, z

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