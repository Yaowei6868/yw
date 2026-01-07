import pickle
import numpy as np

file_path = 'data/Ethereum/raw/ethereum.pkl'

print(f"Loading {file_path}...")
with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(f"数据类型: {type(data)}")

if isinstance(data, dict):
    print("包含的 Keys:", data.keys())
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(f"Key '{k}' shape: {v.shape}")
        else:
            print(f"Key '{k}' value example: {v}")
elif isinstance(data, list):
    print(f"是一个列表，长度: {len(data)}")
    print("第一个元素:", data[0])
else:
    print(data)