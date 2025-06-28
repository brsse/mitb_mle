#!/bin/bash
# model_inf.sh

SNAPSHOT_DATE=$1     # e.g., 2023-08-01
MODEL_ID=$2          # e.g., 20230801

python3 scripts/model_inference.py \
  --snapshotdate "$SNAPSHOT_DATE" \
  --modelpath "model_bank/model_${MODEL_ID}.pkl"