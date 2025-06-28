#!/bin/bash

echo "🔧 Fixing gold data for model training..."
echo "=========================================="

# Run gold processing for the training window (2023-01-01 to 2024-01-01)
echo "📅 Processing gold data for training window: 2023-01-01 to 2024-01-01"

cd /opt/airflow/scripts

# Run the gold processing script for the training window
python3 run_gold_for_training.py --startdate 2023-01-01 --enddate 2024-01-01

echo "✅ Gold data processing complete!"
echo "🎯 You can now run model training again." 