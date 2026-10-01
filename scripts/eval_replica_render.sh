#!/bin/bash

set -euo pipefail

# Render-eval driver, mirrors scripts/eval_replica_all_scans.sh.

GT_MODE="image"              # image | mesh
RENDER_MODE="emission"          # light | emission
ENGINE="CYCLES"              # CYCLES | EEVEE
SAMPLES=64
DEVICE="GPU"                 # GPU | CPU
TEST_EVERY=8
DATASET_ROOT="datasets/Replica"
EXP_ROOT="exps/Replica"
EXPS_OVERVIEW_ROOT="$EXP_ROOT"
TARGET_SCANS=("scan1")
UPDATE_OVERVIEW="true"
SKIP_RENDER="false"

if [[ "$GT_MODE" != "image" && "$GT_MODE" != "mesh" ]]; then
    echo "[ERROR] GT_MODE must be image or mesh"
    exit 1
fi
if [[ "$RENDER_MODE" != "light" && "$RENDER_MODE" != "emission" ]]; then
    echo "[ERROR] RENDER_MODE must be light or emission"
    exit 1
fi

UPDATE_FLAG="--update_overview"
[[ "$UPDATE_OVERVIEW" == "false" ]] && UPDATE_FLAG="--no_update_overview"

SKIP_FLAG=""
[[ "$SKIP_RENDER" == "true" ]] && SKIP_FLAG="--skip_render"

SCAN_DIRS=()
if [[ ${#TARGET_SCANS[@]} -gt 0 ]]; then
    for scan_name in "${TARGET_SCANS[@]}"; do
        scan_path="$DATASET_ROOT/$scan_name"
        if [[ -d "$scan_path" ]]; then
            SCAN_DIRS+=("$scan_path")
        else
            echo "[WARN] target scan not found, skip: $scan_path"
        fi
    done
else
    mapfile -t SCAN_DIRS < <(find "$DATASET_ROOT" -maxdepth 1 -type d -name 'scan*' | sort -V)
fi

if [[ ${#SCAN_DIRS[@]} -eq 0 ]]; then
    echo "[ERROR] no scan dirs under: $DATASET_ROOT"
    exit 1
fi

echo "========================================"
echo "Replica render-eval start"
echo "gt_mode      : $GT_MODE"
echo "render_mode  : $RENDER_MODE"
echo "engine       : $ENGINE   samples=$SAMPLES   device=$DEVICE"
echo "dataset_root : $DATASET_ROOT"
echo "exp_root     : $EXP_ROOT"
echo "scan_count   : ${#SCAN_DIRS[@]}"
echo "========================================"

OK=0
SKIP=0

for scan_path in "${SCAN_DIRS[@]}"; do
    scan_name="$(basename "$scan_path")"
    exp_path="$EXP_ROOT/$scan_name"

    if [[ ! -d "$exp_path" ]]; then
        echo "[SKIP] $scan_name (missing exp dir: $exp_path)"
        SKIP=$((SKIP + 1))
        continue
    fi
    if [[ ! -f "$exp_path/scene_combined.glb" ]]; then
        echo "[SKIP] $scan_name (missing scene_combined.glb)"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo "----------------------------------------"
    echo "[RUN ] $scan_name"

    python evaluation/eval_render_replica.py \
        --scan_path "$scan_path" \
        --exp_path "$exp_path" \
        --gt_mode "$GT_MODE" \
        --render_mode "$RENDER_MODE" \
        --engine "$ENGINE" \
        --samples "$SAMPLES" \
        --device "$DEVICE" \
        --test_every "$TEST_EVERY" \
        --exps_root "$EXPS_OVERVIEW_ROOT" \
        --no_add_floor \
        $SKIP_FLAG \
        "$UPDATE_FLAG"

    echo "[DONE] $scan_name"
    OK=$((OK + 1))
done

echo "========================================"
echo "Replica render-eval finished"
echo "done   : $OK"
echo "skipped: $SKIP"
echo "========================================"
