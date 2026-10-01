#!/usr/bin/env bash

set -euo pipefail

export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Environment for DecomVoxel
source "$REPO_ROOT/env.sh"

CUDA_MAJOR_VERSION="$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1)"
if [[ "$CUDA_MAJOR_VERSION" != "12" ]]; then
    echo "[setup] CUDA 12.x is required by the pinned PyTorch cu124 build; found CUDA ${CUDA_MAJOR_VERSION:-unknown} at $CUDA_HOME." >&2
    echo "[setup] Set CUDA_HOME to a CUDA 12.x toolkit and run setup.sh again." >&2
    exit 1
fi

# Base
pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
pip install plotly==6.0.1 blenderproc kornia pyexr psutil
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/

# TRELLIS
pushd "$REPO_ROOT/decomvoxel/model/TRELLIS" >/dev/null
bash ./setup_trellis.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
popd >/dev/null
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
pip install git+https://github.com/jukgei/diff-gaussian-rasterization.git@b1e1cb83e27923579983a9ed19640c6031112b94 --no-build-isolation
pip install git+https://github.com/openai/CLIP.git
pip install lpips open3d OmegaConf spaces pyexr gradio_litmodel3d multipledispatch loguru mathutils open_clip_torch
pip install gradio==5.34.2
pip install diffusers
pip install ninja
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install flash_attn==2.7.4.post1
pip install kaleido==0.2.1

# GeoSVR
pip install mkl==2023.1.0
pip install git+https://github.com/rahul-goel/fused-ssim.git@3006269823fc28110ba44686a172cbd59ec01bc3 --no-build-isolation
pip install "$REPO_ROOT/decomvoxel/representation/GeoSVR/cuda" --no-build-isolation
pip install yacs natsort argparse imageio imageio-ffmpeg plyfile shapely trimesh==4.0.4 open3d==0.18.0 gpytoolbox lpips pytorch-msssim

# Miscellaneous
pip install rembg
pip install "numpy<2.0"
pip uninstall -y kaolin && pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
pip install pyvista
pip install pymeshfix
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install --upgrade "openai>=1.0"

pip install flash-linear-attention==0.1.1 --no-deps
python -c "from fla.ops.gla import chunk_gla; print('✅ fla successfully import！')"

pip install xformers==0.0.28.post2 --no-deps --index-url https://download.pytorch.org/whl/cu124

pip install accelerate --no-deps
pip install git+https://github.com/facebookresearch/segment-anything.git


# Test TRELLIS
python -c "
import sys
import os
sys.path.insert(0, os.path.join(os.environ['REPO_ROOT'], 'decomvoxel/model/TRELLIS'))
try:
    import trellis.models
    from trellis.pipelines import TrellisImageTo3DPipeline
    print('✓ Import successful!')
    print('✓ TRELLIS models and pipelines loaded')
except Exception as e:
    print(f'✗ Import failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"


# evaluation
pip install pyrender --no-deps
pip install piq
pip install openpyxl 
