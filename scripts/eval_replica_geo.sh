#!/bin/bash

set -euo pipefail

MODE="objects_only"
DATASET_ROOT="datasets/Replica"
EXP_ROOT="exps/Replica"
TARGET_SCANS=("scan1")
UPDATE_OVERVIEW="true"


if [[ "$MODE" != "merged" && "$MODE" != "objects_only" && "$MODE" != "background_only" ]]; then
    echo "[ERROR] mode must be one of: merged, objects_only, background_only"
    exit 1
fi

if [[ "$UPDATE_OVERVIEW" != "true" && "$UPDATE_OVERVIEW" != "false" ]]; then
    echo "[ERROR] UPDATE_OVERVIEW must be true or false"
    exit 1
fi

UPDATE_FLAG="--update_overview"
if [[ "$UPDATE_OVERVIEW" == "false" ]]; then
    UPDATE_FLAG="--no_update_overview"
fi

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
    echo "[ERROR] no scan dirs found under: $DATASET_ROOT"
    exit 1
fi

echo "========================================"
echo "Replica eval start"
echo "mode: $MODE"
echo "dataset_root: $DATASET_ROOT"
echo "exp_root: $EXP_ROOT"
echo "update_overview: $UPDATE_OVERVIEW"
echo "scan_count: ${#SCAN_DIRS[@]}"
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

    echo "----------------------------------------"
    echo "[RUN ] $scan_name"

    python evaluation/eval_geo_replica.py \
        --scan_path "$scan_path" \
        --exp_path "$exp_path" \
        --mode "$MODE" \
        --exps_replica_root "$EXP_ROOT" \
        "$UPDATE_FLAG"

    echo "[DONE] $scan_name"
    OK=$((OK + 1))
done

echo "========================================"
echo "Replica eval finished"
echo "done: $OK"
echo "skipped: $SKIP"
echo "========================================"
