#!/bin/bash
SNAPSHOT_DATE=$1
MODEL_NAME="xgb_model"  # or pass as $2

cd /opt/airflow/scripts
python3 model_monitoring.py \
    --snapshotdate $SNAPSHOT_DATE \
    --modelname $MODEL_NAME