#!/bin/bash

python scr/opt/airflow/scripts/model_training.py\
    --snapshotdate "$1" \
    --modelname "$2"