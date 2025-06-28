# scripts/run_gold.py
import argparse
from pyspark.sql import SparkSession
from gold_processing import build_feature_store, build_label_store

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="feature_store or label_store")
    parser.add_argument("--snapshotdate", type=str, required=True, help="Snapshot date in YYYY-MM-DD format")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("Gold Processing").getOrCreate()

    if args.task == "feature_store":
        build_feature_store(spark, args.snapshotdate)
    elif args.task == "label_store":
        build_label_store(spark, args.snapshotdate)
    else:
        raise ValueError("Unsupported task type. Use 'feature_store' or 'label_store'")