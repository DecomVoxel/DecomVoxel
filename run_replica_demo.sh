#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

EXP="${EXP:-outputs/demo_replica}"
CFG_FILES="decomvoxel/representation/GeoSVR/cfg/mipnerf360_mesh.yaml"
CONFIG="configs/replica_config.yaml"
NUM_VIEWS="${NUM_VIEWS:-200}"
PARALLEL_NUM="${PARALLEL_NUM:-3}"

# Respect CUDA_VISIBLE_DEVICES when it is already set. GPU can be used as a
# convenient explicit override; otherwise train_demo.py selects a free GPU.
if [[ -n "${GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

python train_demo.py --mode full_pipeline \
    --source_path datasets/Replica/scan1 \
    --model_path "${EXP}/scan1" \
    --cfg_files "$CFG_FILES" \
    --config "$CONFIG" \
    --num_views "$NUM_VIEWS" \
    --parallel_num "$PARALLEL_NUM"
