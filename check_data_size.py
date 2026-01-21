import torch
import os

# 1. 加载处理好的数据
data_path = 'data/elliptic++actor/processed/elliptic_plus_actor.pt'
if not os.path.exists(data_path):
    print(f"❌ 文件不存在: {data_path}")
    print("请先运行 force_process.py")
    exit()

print(f"正在加载 {data_path} ...")
data_list, _ = torch.load(data_path)
data = data_list[0]

# 2. 打印关键维度
print("\n=== 数据集维度诊断 ===")
print(f"节点数量 (x.shape[0]): {data.x.shape[0]}")
print(f"特征维度 (x.shape[1]): {data.x.shape[1]} <--- 重点检查这里！正常应为 56 左右")
print(f"边数量 (edge_index.shape[1]): {data.edge_index.shape[1]}")
print(f"标签数量: {data.y.shape[0]}")
print(f"时间步范围: {data.timesteps.min()} ~ {data.timesteps.max()}")

# 3. 估算显存占用
x_mem = data.x.element_size() * data.x.nelement() / 1024**3
edge_mem = data.edge_index.element_size() * data.edge_index.nelement() / 1024**3
print(f"\n估算显存占用:")
print(f"特征矩阵 x: {x_mem:.4f} GB")
print(f"边索引 edge_index: {edge_mem:.4f} GB")

if x_mem > 5.0:
    print("\n⚠️ 警告: 特征矩阵异常巨大！可能是读取了错误的 CSV 列。")
else:
    print("\n✅ 数据尺寸看起来正常。可能是模型中间变量占用了显存。")