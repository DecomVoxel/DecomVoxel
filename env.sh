#!/usr/bin/env bash

# Keep the caller's active Conda environment. The README activates
# `decomvoxel` before sourcing this file, so no installation-specific Conda
# path is required here.
if [[ "${CONDA_DEFAULT_ENV:-}" != "decomvoxel" ]]; then
    echo "[env] Warning: expected the 'decomvoxel' Conda environment to be active."
fi

# Respect an explicitly configured CUDA_HOME. Otherwise derive it from nvcc so
# the repository is not tied to a particular /usr/local/cuda-* installation.
if [[ -z "${CUDA_HOME:-}" ]]; then
    DECOMVOXEL_NVCC="$(command -v nvcc 2>/dev/null || true)"
    DECOMVOXEL_NVCC_MAJOR=""
    if [[ -n "$DECOMVOXEL_NVCC" ]]; then
        DECOMVOXEL_NVCC_MAJOR="$("$DECOMVOXEL_NVCC" --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1)"
    fi

    if [[ "$DECOMVOXEL_NVCC_MAJOR" == "12" ]]; then
        export CUDA_HOME="$(cd "$(dirname "$DECOMVOXEL_NVCC")/.." && pwd)"
    else
        for DECOMVOXEL_CUDA_CANDIDATE in /usr/local/cuda-12.* /usr/local/cuda-12; do
            if [[ -x "$DECOMVOXEL_CUDA_CANDIDATE/bin/nvcc" ]]; then
                export CUDA_HOME="$DECOMVOXEL_CUDA_CANDIDATE"
                break
            fi
        done
        if [[ -z "${CUDA_HOME:-}" && -n "$DECOMVOXEL_NVCC" ]]; then
            export CUDA_HOME="$(cd "$(dirname "$DECOMVOXEL_NVCC")/.." && pwd)"
        fi
    fi
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
    export CUDA_INCLUDE="$CUDA_HOME/include"
    export CUDA_LIB="$CUDA_HOME/lib64"
    export PATH="$CUDA_HOME/bin:${PATH:-}"
    export LD_LIBRARY_PATH="$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="$CUDA_LIB${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export CFLAGS="-I$CUDA_HOME/include${CFLAGS:+ $CFLAGS}"
    export CXXFLAGS="-I$CUDA_HOME/include${CXXFLAGS:+ $CXXFLAGS}"
    export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
    echo "CUDA_HOME: $CUDA_HOME"
    "$CUDA_HOME/bin/nvcc" --version
else
    echo "[env] Warning: nvcc was not found. Set CUDA_HOME before building CUDA extensions."
fi

export PATH="${PATH:-}:$HOME/.local/bin"

# Preserve keys already exported by the user. They may also be filled in here.
export ARK_API_KEY="${ARK_API_KEY:-}"
export ATLASCLOUD_API_KEY="${ATLASCLOUD_API_KEY:-}"
