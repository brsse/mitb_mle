import argparse
import os
import pickle
import pandas as pd
import pyspark
from pyspark.sql.functions import col
from datetime import datetime

def main(snapshotdate, modelname):
    print(f"📦 Running inference for snapshot: {snapshotdate} using model: {modelname}")

    # Load Spark
    spark = pyspark.sql.SparkSession.builder \
        .appName("InferenceJob") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load model artefact
    model_path = f"model_bank/{modelname}"
    with open(model_path, 'rb') as f:
        model_artefact = pickle.load(f)

    model = model_artefact["model"]
    transformer = model_artefact["preprocessing_transformers"]["stdscaler"]

    # Load feature store snapshot
    feature_path = "datamart/gold/feature_store"
    features_sdf = spark.read.parquet(feature_path).filter(col("snapshot_date") == snapshotdate)

    features_pdf = features_sdf.toPandas()
    feature_cols = [c for c in features_pdf.columns if c.startswith("fe_")]

    X = transformer.transform(features_pdf[feature_cols])
    y_proba = model.predict_proba(X)[:, 1]

    # Prepare output
    output_pdf = features_pdf[["Customer_ID", "snapshot_date"]].copy()
    output_pdf["model_name"] = modelname
    output_pdf["model_prediction"] = y_proba

    # Save output
    model_id = modelname.replace(".pkl", "")
    output_dir = f"datamart/model_predictions/{model_id}/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{model_id}_predictions_{snapshotdate.replace('-', '_')}.parquet"

    spark.createDataFrame(output_pdf).write.mode("overwrite").parquet(os.path.join(output_dir, output_file))
    print(f"✅ Inference complete. Results saved to: {os.path.join(output_dir, output_file)}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference.")
    parser.add_argument("--snapshotdate", type=str, required=True, help="e.g. 2024-09-01")
    parser.add_argument("--modelname", type=str, required=True, help="e.g. xgb_v1.pkl")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)