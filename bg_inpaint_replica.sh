#!/bin/bash
# bg_inpaint_replica_para.sh
#
# Entry point: bg-inpaint pipeline for Replica scan1 only, with end-to-end
# timing and background GPU monitoring.
#
# Timing + GPU logs -> outputs/bg_replica_efficiency/scan1/log/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash bg_inpaint_replica_para.sh
# Or override GPU / paths via environment:
#   GPU=4 bash bg_inpaint_replica_para.sh

set -ue
export CUDA_DEVICE_ORDER=PCI_BUS_ID

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── Configurable knobs ────────────────────────────────────────────────────────
SCENE="${SCENE:-scan1}"
# GPU precedence:
#   1) explicit GPU env var
#   2) first entry from CUDA_VISIBLE_DEVICES
#   3) default 0
if [[ -n "${GPU:-}" ]]; then
    GPU="${GPU}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU="${CUDA_VISIBLE_DEVICES%%,*}"
else
    GPU="0"
fi
CKPT_ROOT="${CKPT_ROOT:-outputs/demo_replica}"
OUT_ROOT="${OUT_ROOT:-outputs/bg}"
EXP_NAME="${EXP_NAME:-bg_training}"
GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-10.0}"

export CUDA_VISIBLE_DEVICES="$GPU"

echo "===== bg_inpaint_replica_para.sh ====="
echo "  scene               : $SCENE"
echo "  GPU                 : $GPU"
echo "  ckpt_root           : $CKPT_ROOT"
echo "  out_root            : $OUT_ROOT"
echo "  exp_name            : $EXP_NAME"
echo "  gpu_sample_interval : ${GPU_SAMPLE_INTERVAL}s"
echo "======================================="

exec python scripts/bg_inpaint_efficiency_driver.py \
    --scene               "$SCENE" \
    --gpu                 "$GPU" \
    --ckpt_root           "$CKPT_ROOT" \
    --out_root            "$OUT_ROOT" \
    --exp_name            "$EXP_NAME" \
    --gpu_sample_interval "$GPU_SAMPLE_INTERVAL"
