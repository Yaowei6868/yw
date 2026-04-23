#!/bin/bash
# setup_cloud_env.sh
# 在 Linux 云服务器上创建并初始化本项目运行环境。
#
# 默认假设:
# - Ubuntu / Debian 系
# - Python 3.10+
# - 使用 venv
#
# 可选环境变量:
#   PYTHON_BIN=python3.10
#   VENV_DIR=.venv
#   TORCH_VERSION=2.5.1
#   CUDA_TAG=cu121        # 可选: cu121 / cpu
#
# 用法:
#   bash scripts/setup_cloud_env.sh
#   CUDA_TAG=cpu bash scripts/setup_cloud_env.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
CUDA_TAG="${CUDA_TAG:-cu121}"

echo "[INFO] Root dir   : ${ROOT_DIR}"
echo "[INFO] Python bin : ${PYTHON_BIN}"
echo "[INFO] Venv dir   : ${VENV_DIR}"
echo "[INFO] Torch ver  : ${TORCH_VERSION}"
echo "[INFO] CUDA tag   : ${CUDA_TAG}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: ${PYTHON_BIN}"
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "[INFO] Creating virtual environment..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [ "${CUDA_TAG}" = "cpu" ]; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
  PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html"
else
  TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
  PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"
fi

echo "[INFO] Installing PyTorch..."
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==0.20.1" \
  "torchaudio==2.5.1" \
  --index-url "${TORCH_INDEX_URL}"

echo "[INFO] Installing project Python dependencies..."
python -m pip install \
  numpy \
  matplotlib \
  pandas \
  scikit-learn \
  networkx \
  omegaconf \
  tensorboard

echo "[INFO] Installing PyG stack..."
python -m pip install \
  pyg_lib \
  torch_scatter \
  torch_sparse \
  torch_cluster \
  torch_spline_conv \
  -f "${PYG_WHEEL_URL}"

python -m pip install torch-geometric

echo "[INFO] Running environment check..."
python "${ROOT_DIR}/tools/check_env.py"

echo ""
echo "[DONE] Cloud environment is ready."
echo "Activate with:"
echo "  source \"${VENV_DIR}/bin/activate\""
echo ""
echo "Example run:"
echo "  python train.py --config configs/ours/main/elliptic_TASDCL_CGNN.yaml"
