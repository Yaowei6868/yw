import dgl
from dgl.data.utils import load_graphs
import torch

def analyze_features():
    print("正在加载 T-Finance 数据集...")
    # 指向你放置原始二进制文件的路径
    g_list, _ = load_graphs('data/tfinance/raw/tfinance/tfinance')
    g = g_list[0]
    x = g.ndata['feature']
    
    print(f"特征矩阵加载成功，形状: {x.shape}")
    print("=" * 60)
    
    # 逐列分析 10 维特征
    for i in range(x.shape[1]):
        col = x[:, i]
        num_uniques = len(torch.unique(col))
        min_val = col.min().item()
        max_val = col.max().item()
        mean_val = col.mean().item()
        std_val = col.std().item()
        
        # 检查是否全部是整数（或者非常接近整数，排除浮点误差）
        is_integer = torch.allclose(col, col.round(), atol=1e-4)
        
        print(f"【第 {i} 列特征分析】:")
        print(f"  -> 唯一值数量: {num_uniques}")
        print(f"  -> 最小值: {min_val:.4f}, 最大值: {max_val:.4f}")
        print(f"  -> 均值: {mean_val:.4f}, 标准差: {std_val:.4f}")
        print(f"  -> 是否全为整数 (离散天数特征标志): {is_integer}")
        print("-" * 60)

if __name__ == '__main__':
    analyze_features()