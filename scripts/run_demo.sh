#!/bin/bash
# Run all demo scenes in parallel, one GPU per scene.
#
# Usage:
#   bash scripts/run_demo.sh
#
# GPU assignment (edit as needed):
#   berlin   -> GPU 0
#   mipnerf  -> GPU 1
#   playroom -> GPU 3
#   nyc      -> GPU 4
#   grocery  -> GPU 5

set -u

EXP="outputs/exp_demo"
CFG_FILES="decomvoxel/representation/GeoSVR/cfg/mipnerf360_mesh.yaml"
CONFIG="configs/sds_config_uncertainty-13.yaml"
NUM_VIEWS=60

# CUDA_VISIBLE_DEVICES=0 python train_geo_sds_scene.py --mode full_pipeline \
#     --source_path datasets/demo/berlin \
#     --model_path ${EXP}/berlin \
#     --cfg_files ${CFG_FILES} \
#     --config ${CONFIG} \
#     --num_views ${NUM_VIEWS} &
# PID_BERLIN=$!

# CUDA_VISIBLE_DEVICES=1 python train_geo_sds_scene.py --mode full_pipeline \
#     --source_path datasets/demo/mipnerf \
#     --model_path ${EXP}/mipnerf \
#     --cfg_files ${CFG_FILES} \
#     --config ${CONFIG} \
#     --num_views ${NUM_VIEWS} &
# PID_MIPNERF=$!

# CUDA_VISIBLE_DEVICES=3 python train_geo_sds_scene.py --mode full_pipeline \
#     --source_path datasets/demo/playroom \
#     --model_path ${EXP}/playroom \
#     --cfg_files ${CFG_FILES} \
#     --config ${CONFIG} \
#     --num_views ${NUM_VIEWS} &
# PID_PLAYROOM=$!

# CUDA_VISIBLE_DEVICES=4 python train_geo_sds_scene.py --mode full_pipeline \
#     --source_path datasets/demo/nyc \
#     --model_path ${EXP}/nyc \
#     --cfg_files ${CFG_FILES} \
#     --config ${CONFIG} \
#     --num_views ${NUM_VIEWS} &
# PID_NYC=$!

if [[ -n "${GPU_GROCERY:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_GROCERY"
fi
python train_geo_sds_scene.py --mode full_pipeline \
    --source_path datasets/demo/grocery \
    --model_path ${EXP}/grocery \
    --cfg_files ${CFG_FILES} \
    --config ${CONFIG} \
    --num_views ${NUM_VIEWS} &
PID_GROCERY=$!

wait "$PID_BERLIN"   && echo "[done] berlin"   || echo "[FAIL] berlin"
wait "$PID_MIPNERF"  && echo "[done] mipnerf"  || echo "[FAIL] mipnerf"
wait "$PID_PLAYROOM" && echo "[done] playroom"  || echo "[FAIL] playroom"
wait "$PID_NYC"      && echo "[done] nyc"       || echo "[FAIL] nyc"
wait "$PID_GROCERY"  && echo "[done] grocery"   || echo "[FAIL] grocery"

echo "All demo scenes finished."
