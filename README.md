
# fraud-detection-gnn

图欺诈检测与 task-only continual learning 实验代码库。

## Cloud Setup

推荐云端环境：

- Linux x86_64
- Python 3.10
- CUDA 12.1
- 单独虚拟环境

快速部署：

```bash
bash scripts/setup_cloud_env.sh
```

CPU-only 环境：

```bash
CUDA_TAG=cpu bash scripts/setup_cloud_env.sh
```

部署完成后验证环境：

```bash
source .venv/bin/activate
python tools/check_env.py
```

## Main Entry

```bash
python train.py --config configs/ours/main/elliptic_TASDCL_CGNN.yaml
```

## Useful Runs

Elliptic CGNN strict ablation:

```bash
bash scripts/run_elliptic_cgnn_ablation.sh
```

Elliptic CL baselines:

```bash
bash scripts/run_elliptic_cl_baselines.sh
```

## Notes

- `requirements.txt` 仅保留通用 Python 依赖；`torch` 与 `torch_geometric` 的安装优先以云端脚本为准。
- DGraphFin 需要手动下载原始压缩包到 `data/dgraphfin/raw/`。
- 当前主实验配置默认使用 `cuda` 设备。
