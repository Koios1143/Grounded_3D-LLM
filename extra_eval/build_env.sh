#!/bin/bash
# Grounded 3D-LLM inference env as a uv venv, ported to current GPUs: the release pins
# torch 1.12.1+cu116, which has no kernels for Blackwell (sm_120), so this builds
# torch 2.7.1+cu128 + MinkowskiEngine (patched for CUDA 12 / C++17) + pointnet2 ops.
#   bash extra_eval/build_env.sh
# Run on a GPU node with internet access (compiles CUDA code). Paths come from env.sh:
#   G3D_ENV     where the venv goes                       (default .venv)
#   G3D_ROOT    scratch for the MinkowskiEngine checkout   (default .)
#   G3D_CUDA_HOME  CUDA 12.x toolkit, >= 12.8               (default /usr/local/cuda)
#   TORCH_CUDA_ARCH_LIST  GPU archs to compile for         (default: the visible GPU's)
# Needs an OpenBLAS with cblas.h (Ubuntu: libopenblas-dev) for MinkowskiEngine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
UV="${UV:-uv}"
ENV="${G3D_ENV:-.venv}"
export CUDA_HOME="${G3D_CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-16}"

$UV python install 3.10
[ -x "$ENV/bin/python" ] || $UV venv "$ENV" -p 3.10 --seed
PY="$ENV/bin/python"
export PATH="$(cd "$ENV" && pwd)/bin:$PATH"  # MinkowskiEngine's setup.py calls a bare `pip`
PIP="$UV pip install --python $PY"
$PIP torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
$PIP torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
$PIP -r extra_eval/requirements.txt --override extra_eval/overrides.txt
$PIP peft==0.8.2 --no-deps
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$($PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')}"
echo "CUDA_HOME=$CUDA_HOME TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# MinkowskiEngine at the commit the authors used + minkowski_cuda12.patch (CUDA 12.9 / GCC 13:
# missing thrust includes, NVTX3 clash with CUDA's bundled copy, shared_ptr/__to_address).
ME_DIR="${G3D_ROOT:-.}/build/MinkowskiEngine"
if ! $PY -c "import MinkowskiEngine" 2>/dev/null; then
    [ -d "$ME_DIR" ] || git clone --recursive https://github.com/NVIDIA/MinkowskiEngine "$ME_DIR"
    cd "$ME_DIR"
    git checkout -f 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
    git apply "$REPO/extra_eval/minkowski_cuda12.patch"
    BLAS_INC="${BLAS_INCLUDE_DIR:-/usr/include/x86_64-linux-gnu}"
    $PY setup.py install --blas_include_dirs="$BLAS_INC" --blas=openblas --force_cuda
    cd "$REPO"
fi

# pointnet2 ops (installed as top-level package `pointnet2._ext`)
cd third_party/pointnet2
$PY setup.py install
rm -rf build dist *.egg-info
cd "$REPO"

PYTHONPATH=. $PY -c "
import torch, MinkowskiEngine as ME, torch_scatter, transformers, peft, pytorch_lightning as pl
import pointnet2._ext
print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('ME', ME.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'pl', pl.__version__)
from trainer.trainer import ModelingGrounded3DLLM; import datasets.semseg; print('grounded3dllm imports ok')
"
echo ENV_DONE
