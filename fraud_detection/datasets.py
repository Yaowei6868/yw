import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from torch_geometric.data import Data,InMemoryDataset
from sklearn.model_selection import train_test_split


class EllipticPlusActorDataset(InMemoryDataset):
    def __init__(self, root='data/elliptic++actor', transform=None, pre_transform=None):
        super(EllipticPlusActorDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # 确保您的文件名和这里一致
        return ['wallets_features.csv', 'wallets_classes.csv', 'AddrAddr_edgelist.csv']

    @property
    def processed_file_names(self):
        return ['elliptic_plus_actor.pt']

    def process(self):
        print("正在处理 Elliptic++ Actor 数据集，这可能需要几分钟...")
        
        # 1. 定义路径
        feat_path = os.path.join(self.raw_dir, 'wallets_features.csv')
        label_path = os.path.join(self.raw_dir, 'wallets_classes.csv')
        edge_path = os.path.join(self.raw_dir, 'AddrAddr_edgelist.csv')

        # 2. 读取 CSV (使用 pandas)
        # 注意：Elliptic++ 的 CSV 通常包含表头，如果没有请去掉 header=0
        print("正在读取 CSV 文件...")
        df_feat = pd.read_csv(feat_path)
        df_label = pd.read_csv(label_path)
        df_edge = pd.read_csv(edge_path)
        
        # 3. 构建节点映射 (Address String -> Index Int)
        # 因为钱包地址是字符串，我们需要给它们编个号
        print("正在构建节点索引映射...")
        # 假设第一列是 address
        all_nodes = df_feat.iloc[:, 0].unique() # 获取所有钱包地址
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        num_nodes = len(all_nodes)
        print(f"总节点数: {num_nodes}")

        # 4. 处理特征 (Features)
        print("处理特征...")
        # 假设第一列是地址，后面全是特征
        # drop 第一列 (address)，保留数值特征
        feat_values = df_feat.iloc[:, 1:].values 
        # 处理可能的 NaN (填充为 0)
        feat_values = np.nan_to_num(feat_values.astype(np.float32))
        x = torch.tensor(feat_values, dtype=torch.float)

        # 5. 处理标签 (Labels)
        print("处理标签...")
        # 假设 label 文件列名是: address, class
        # 映射 label: 1(Illicit)->1, 2(Licit)->0, 3/Unknown -> -1
        # 先把 label df 对齐到 node_to_idx 的顺序
        # 这一步比较关键，要保证顺序一致
        df_label = df_label.set_index(df_label.columns[0]) # 设地址为索引
        labels_aligned = df_label.reindex(all_nodes)['class'].values # 按 all_nodes 顺序取 label
        
        y = torch.full((num_nodes,), -1, dtype=torch.long) # 默认为 -1 (未知)
        
        # 填入标签
        # 注意：Elliptic++ 的 class 定义通常是 1=Illicit, 2=Licit, 3=Unknown
        # 我们做异常检测：Illicit(1) -> 1, Licit(2) -> 0
        y[labels_aligned == 1] = 1 # 坏人
        y[labels_aligned == 2] = 0 # 好人
        # 其他的保持 -1 (在训练时会被 mask 掉)
        
        print(f"非法节点数 (Class 1): {(y==1).sum().item()}")
        print(f"合法节点数 (Class 0): {(y==0).sum().item()}")

        # 6. 处理边 (Edges)
        print("处理边...")
        # 将源地址和目标地址 转换为 索引
        src_list = df_edge.iloc[:, 0].map(node_to_idx).dropna().astype(int).values
        dst_list = df_edge.iloc[:, 1].map(node_to_idx).dropna().astype(int).values
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # 7. 生成时间步 (Timesteps)
        # 增量学习必须项。Elliptic++ Actor 是静态聚合图，没有天然时间步。
        # 我们模拟生成：将数据均匀切分成 10 份 (Task 0 - 9)
        print("生成模拟时间步 (用于增量学习)...")
        # 简单策略：直接按节点顺序切分 (因为通常数据是按时间顺序采集的)
        # 或者随机切分：timesteps = torch.randint(0, 10, (num_nodes,))
        # 这里用均匀切分：
        steps = 10
        nodes_per_step = (num_nodes // steps) + 1
        timesteps = torch.arange(num_nodes) // nodes_per_step
        timesteps = timesteps.clamp(max=steps-1).long()

        # 8. 保存数据
        data = Data(x=x, edge_index=edge_index, y=y)
        data.timesteps = timesteps
        
        # 只有两个mask：已分类(trainable) 和 未分类
        # 只有 y != -1 的节点才应该进入 mask
        labeled_mask = (y != -1)
        # 把有标签的节点索引拿出来
        labeled_indices = torch.where(labeled_mask)[0]
        
        # 为了代码兼容，我们把有标签的设为 classified
        data.classified_idx = labeled_indices
        data.unclassified_idx = torch.where(~labeled_mask)[0]

        if self.pre_filter is not None and not self.pre_filter(data):
            pass
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
        print("✅ 处理完成！elliptic_plus_actor.pt 已保存。")

class EllipticDataset:
    def __init__(self, config):
        # 1. 加载原始数据
        self.features_df = pd.read_csv(config.features_path, header=None)
        self.edges_df = pd.read_csv(config.edges_path)
        self.labels_df = pd.read_csv(config.classes)
        
        # 2. 标签映射: 'unknown'->2, '1'(非法)->1, '2'(合法)->0
        self.labels_df["class"] = self.labels_df["class"].map({'unknown': 2, '1': 1, '2': 0})
        
        # 3. 合并数据
        self.merged_df = self.merge()
        
        # 4. 构建图数据组件
        self.edge_index = self._edge_index()
        self.edge_weights = self._edge_weights()
        
        # [关键修改] 这里生成的特征将只包含前 94 维
        self.node_features = self._node_features()
        
        self.labels = self._labels()
        self.classified_ids = self._classified_ids()
        self.unclassified_ids = self._unclassified_ids()
        self.licit_ids = self._licit_ids()
        self.illicit_ids = self._illicit_ids()

    def visualize_distribution(self):
        groups = self.labels_df.groupby("class").count()
        plt.title("Classes distribution")
        plt.barh(['Licit', 'Illicit', 'Unknown'], groups['txId'].values, color=['green', 'red', 'grey'])
        plt.show()

    def merge(self):
        # 合并特征表和标签表，并按 ID 排序
        df_merge = self.features_df.merge(self.labels_df, how='left', right_on="txId", left_on=0)
        df_merge = df_merge.sort_values(0).reset_index(drop=True)
        return df_merge

    def train_test_split(self, test_size=0.15):
        # 辅助方法，Trainer 可根据需要调用
        train_idx, valid_idx = train_test_split(self.classified_ids.values, test_size=test_size)
        return train_idx, valid_idx

    def pyg_dataset(self):
        """
        构建 PyG Data 对象。
        已适配持续学习 (CL)：不进行静态划分，而是提供时间步信息供 Trainer 动态切分。
        """
        dataset = Data(
            x=self.node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_weights,
            y=self.labels,
        )
        
        # [CL 支持] 添加时间步信息 (merged_df 第1列是 time_step)
        dataset.timesteps = torch.tensor(self.merged_df[1].values, dtype=torch.long)
        
        # [CL 支持] 提供所有已标注节点的索引池
        dataset.classified_idx = torch.tensor(self.classified_ids.values, dtype=torch.long)
        dataset.unclassified_idx = torch.tensor(self.unclassified_ids.values, dtype=torch.long)
        
        # 兼容性设置
        dataset.test_idx = dataset.unclassified_idx 

        return dataset

    def _licit_ids(self):
        # class == 0
        return self.merged_df[self.merged_df['class'] == 0].index

    def _illicit_ids(self):
        # class == 1
        return self.merged_df[self.merged_df['class'] == 1].index

    def _classified_ids(self):
        # class != 2 (即 0 或 1)
        return self.merged_df[self.merged_df['class'] != 2].index

    def _unclassified_ids(self):
        # class == 2
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
        
        # 映射 ID
        edges.txId1 = edges.txId1.map(ids_mapping)
        edges.txId2 = edges.txId2.map(ids_mapping)
        
        # 移除无效边
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