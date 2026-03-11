import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import train_test_split


class TFinanceDataset(InMemoryDataset):
    def __init__(self, root='data/tfinance', transform=None, pre_transform=None, num_tasks=10):
        # 允许自定义增量学习的任务数量（默认切分为 10 个时间步）
        self.num_tasks = num_tasks 
        super(TFinanceDataset, self).__init__(root, transform, pre_transform)
        
        # 安全加载逻辑，与你的 EllipticPlusActorDataset 保持一致
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print("⚠️ 注意: T-Finance 预处理文件不存在，稍后将自动触发 process()。")

    @property
    def raw_file_names(self):
        return ['tfinance'] # 官方开源的 DGL 数据集文件名

    @property
    def processed_file_names(self):
        return ['tfinance_processed.pt']

    def process(self):
        print("正在处理 T-Finance 数据集 (DGL 转 PyG 并划分时间步)...")
        
        # 1. 加载 DGL 格式的原数据
        try:
            from dgl.data.utils import load_graphs
        except ImportError:
            raise ImportError("处理原始 T-Finance 数据集需要安装 DGL。请在终端运行: pip install dgl")

        raw_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"找不到原始文件 {raw_file}，请从原作者 Google Drive 下载并放入该目录。")
            
        g_list, label_dict = load_graphs(raw_file)
        g = g_list[0] # 取出唯一的图

        # 2. 提取节点特征和标签
        x = g.ndata['feature'].float()
        # 原文代码中标签使用了 argmax 将 one-hot 转为 0/1 (0:正常, 1:异常)
        y = g.ndata['label'].argmax(dim=1).long() 

        # 3. 提取并转换边属性为 PyG 的 edge_index
        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=0).long()

        print(f"原始数据加载完成: 节点数 {x.size(0)}, 边数 {edge_index.size(1)}, 异常节点数 {(y==1).sum().item()}")

        # 4. === 核心：时间步划分逻辑 (模拟数据流) ===
        print(f"正在基于 '注册天数' 代理特征将图划分为 {self.num_tasks} 个增量时间步...")
        
        # 假设 T-Finance 的 10 维特征中，第 0 维代表时间相关的单调变量（如账户注册天数）
        # 我们根据该特征对节点进行排序，还原时间先后顺序
        time_proxy_feature = x[:, 0] 
        
        # 获取排序后的节点索引 (从小到大，代表账户注册从早到晚)
        sorted_indices = torch.argsort(time_proxy_feature)
        
        # 初始化时间步张量，默认全为 0
        timesteps = torch.zeros(x.size(0), dtype=torch.long)
        
        # 平均划分节点到各个 task 中
        nodes_per_task = x.size(0) // self.num_tasks
        
        for task_id in range(self.num_tasks):
            start_idx = task_id * nodes_per_task
            # 最后一个 task 包含所有剩余的零头节点
            end_idx = (task_id + 1) * nodes_per_task if task_id < self.num_tasks - 1 else x.size(0)
            
            # 找到属于当前 task 的节点原始 ID，并打上时间步标签
            current_task_nodes = sorted_indices[start_idx:end_idx]
            timesteps[current_task_nodes] = task_id
            
        print(f"时间步划分完成。Task 分布: {torch.bincount(timesteps)}")

        # 5. 构建 PyG Data 对象
        data = Data(x=x, edge_index=edge_index, y=y)
        data.timesteps = timesteps # 你的增量学习框架强依赖这个属性
        
        # T-Finance 是全图标注的，没有 unclassified 的节点
        data.classified_idx = torch.arange(x.size(0))
        data.unclassified_idx = torch.tensor([], dtype=torch.long)

        if self.pre_filter is not None and not self.pre_filter(data):
            pass
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # 6. 保存为 PyG 格式
        torch.save(self.collate([data]), self.processed_paths[0])
        print("✅ 处理完成！tfinance_processed.pt 已保存至 processed 目录。")


class EllipticPlusActorDataset(InMemoryDataset):
    def __init__(self, root='data/elliptic++actor', transform=None, pre_transform=None):
        super(EllipticPlusActorDataset, self).__init__(root, transform, pre_transform)
        # 【修改点 1】安全加载：只有文件存在时才加载，防止首次运行报错
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print("⚠️ 注意: 预处理文件不存在，跳过加载。如果是首次运行，稍后将自动或手动触发 process()。")

    @property
    def raw_file_names(self):
        return ['wallets_features.csv', 'wallets_classes.csv', 'AddrAddr_edgelist.csv']

    @property
    def processed_file_names(self):
        return ['elliptic_plus_actor.pt']

    def process(self):
        # 【修改点 2】完整的处理逻辑，包含正确的时间步映射
        print("正在处理 Elliptic++ Actor 数据集 (修正版)，这可能需要几分钟...")
        
        # 1. 定义路径
        feat_path = os.path.join(self.raw_dir, 'wallets_features.csv')
        label_path = os.path.join(self.raw_dir, 'wallets_classes.csv')
        edge_path = os.path.join(self.raw_dir, 'AddrAddr_edgelist.csv')

        # 2. 读取 CSV
        print(f"正在读取 CSV 文件: {feat_path} ...")
        df_feat = pd.read_csv(feat_path, header=0) 
        df_label = pd.read_csv(label_path, header=0)
        df_edge = pd.read_csv(edge_path, header=0)
        
        # 3. 构建节点映射 (Address String -> Index Int)
        print("正在构建节点索引映射...")
        # Elliptic++ CSV 结构: [0]=address, [1]=time_step, [2:]=features
        all_nodes = df_feat.iloc[:, 0].astype(str).values 
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        num_nodes = len(all_nodes)
        print(f"总节点数: {num_nodes}")

        # 4. 提取真实的时间步 (Time Step Mapping)
        print("提取并映射时间步...")
        time_col_name = None
        for col in df_feat.columns:
            if 'time' in col.lower() and 'step' in col.lower():
                time_col_name = col
                break
        
        if time_col_name:
            print(f"检测到时间列名为: {time_col_name}")
            raw_timesteps = df_feat[time_col_name].values
        else:
            print("警告：未自动检测到时间列名，默认使用第 2 列 (Index 1) 作为时间步。")
            raw_timesteps = df_feat.iloc[:, 1].values

        # 转换为 Tensor 并映射到 10 个 Task (0-9)
        timesteps_tensor = torch.tensor(raw_timesteps, dtype=torch.long)
        # 映射逻辑: 每 5 个原始时间步归为一个 Task
        timesteps = (timesteps_tensor - 1) // 5
        timesteps = timesteps.clamp(0, 9) 
        
        print(f"时间步映射完成。Task 分布: {torch.bincount(timesteps)}")

        # 5. 处理特征 (Features) - 修正特征范围
        print("处理特征...")
        # 真正的特征应该从第 2 列 (Index 2) 开始 (跳过 address 和 time_step)
        feat_values = df_feat.iloc[:, 2:].values 
        feat_values = np.nan_to_num(feat_values.astype(np.float32))
        x = torch.tensor(feat_values, dtype=torch.float)
        print(f"特征矩阵形状: {x.shape}")

        # 6. 处理标签 (Labels)
        print("处理标签...")
        df_label = df_label.set_index(df_label.columns[0]) 
        labels_aligned_series = df_label.reindex(all_nodes)[df_label.columns[0]]
        labels_aligned = labels_aligned_series.values
        
        y = torch.full((num_nodes,), -1, dtype=torch.long)
        
        # 1=Illicit, 2=Licit
        mask_illicit = (labels_aligned == 1) | (labels_aligned == '1')
        y[mask_illicit] = 1 
        mask_licit = (labels_aligned == 2) | (labels_aligned == '2')
        y[mask_licit] = 0 
        
        print(f"非法节点数 (Class 1): {(y==1).sum().item()}")
        print(f"合法节点数 (Class 0): {(y==0).sum().item()}")

        # 7. 处理边 (Edges)
        print("处理边...")
        src_str = df_edge.iloc[:, 0].astype(str)
        dst_str = df_edge.iloc[:, 1].astype(str)
        
        src_idx = src_str.map(node_to_idx)
        dst_idx = dst_str.map(node_to_idx)
        
        valid_edges_mask = (~src_idx.isna()) & (~dst_idx.isna())
        
        if not valid_edges_mask.all():
            print(f"警告: 丢弃了 {(~valid_edges_mask).sum()} 条连接到未知节点的边。")
            
        src_list = src_idx[valid_edges_mask].astype(int).values
        dst_list = dst_idx[valid_edges_mask].astype(int).values
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # 8. 保存数据
        data = Data(x=x, edge_index=edge_index, y=y)
        data.timesteps = timesteps
        
        labeled_mask = (y != -1)
        data.classified_idx = torch.where(labeled_mask)[0]
        data.unclassified_idx = torch.where(~labeled_mask)[0]

        if self.pre_filter is not None and not self.pre_filter(data):
            pass
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
        print("✅ 处理完成！elliptic_plus_actor.pt 已保存。")

# 保留原有的 EllipticDataset 类，防止 train.py 报错
class EllipticDataset:
    def __init__(self, config):
        self.features_df = pd.read_csv(config.features_path, header=None)
        self.edges_df = pd.read_csv(config.edges_path)
        self.labels_df = pd.read_csv(config.classes)
        self.labels_df["class"] = self.labels_df["class"].map({'unknown': 2, '1': 1, '2': 0})
        self.merged_df = self.merge()
        self.edge_index = self._edge_index()
        self.edge_weights = self._edge_weights()
        self.node_features = self._node_features()
        self.labels = self._labels()
        self.classified_ids = self._classified_ids()
        self.unclassified_ids = self._unclassified_ids()
        self.licit_ids = self._licit_ids()
        self.illicit_ids = self._illicit_ids()

    def merge(self):
        df_merge = self.features_df.merge(self.labels_df, how='left', right_on="txId", left_on=0)
        df_merge = df_merge.sort_values(0).reset_index(drop=True)
        return df_merge

    def train_test_split(self, test_size=0.15):
        train_idx, valid_idx = train_test_split(self.classified_ids.values, test_size=test_size)
        return train_idx, valid_idx

    def pyg_dataset(self):
        dataset = Data(
            x=self.node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_weights,
            y=self.labels,
        )
        dataset.timesteps = torch.tensor(self.merged_df[1].values, dtype=torch.long)
        dataset.classified_idx = torch.tensor(self.classified_ids.values, dtype=torch.long)
        dataset.unclassified_idx = torch.tensor(self.unclassified_ids.values, dtype=torch.long)
        dataset.test_idx = dataset.unclassified_idx 
        return dataset

    def _licit_ids(self):
        return self.merged_df[self.merged_df['class'] == 0].index

    def _illicit_ids(self):
        return self.merged_df[self.merged_df['class'] == 1].index

    def _classified_ids(self):
        return self.merged_df[self.merged_df['class'] != 2].index

    def _unclassified_ids(self):
        return self.merged_df[self.merged_df['class'] == 2].index

    def _node_features(self):
        node_features = self.merged_df.drop(['txId'], axis=1).copy()
        node_features = node_features.drop(columns=[0, 1, "class"])
        node_features_t = torch.tensor(node_features.values, dtype=torch.double)
        return node_features_t

    def _edge_index(self):
        node_ids = self.merged_df[0].values
        ids_mapping = {y: x for x, y in enumerate(node_ids)}
        edges = self.edges_df.copy()
        edges.txId1 = edges.txId1.map(ids_mapping)
        edges.txId2 = edges.txId2.map(ids_mapping)
        edges = edges.dropna().astype(int)
        edge_index = np.array(edges.values).T
        edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
        return edge_index

    def _edge_weights(self):
        weights = torch.tensor([1] * self.edge_index.shape[1], dtype=torch.double)
        return weights

    def _labels(self):
        labels = self.merged_df["class"].values
        labels_tensor = torch.tensor(labels, dtype=torch.double)
        return labels_tensor