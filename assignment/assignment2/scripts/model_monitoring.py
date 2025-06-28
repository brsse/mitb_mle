import pandas as pd
import psycopg2
from psycopg2 import OperationalError
from sklearn.metrics import roc_auc_score, log_loss
from datetime import datetime
import argparse
import time

# CONNECT TO POSTGRESQL

def connect_with_retry(db="airflow", user="airflow", password="airflow",
                       host="postgres", port="5432", retries=5, delay=5):
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(
                dbname=db,
                user=user,
                password=password,
                host=host,
                port=port
            )
            print("✅ Connected to PostgreSQL")
            return conn
        except OperationalError as e:
            print(f"⏳ Retry {attempt + 1} - PostgreSQL not ready yet: {e}")
            time.sleep(delay)
    raise Exception("❌ Could not connect to PostgreSQL after multiple retries.")

# METRIC CALCULATION 

def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df["true_label"]
    y_pred_prob = df["pred_prob"]

    return {
        "inference_count": len(df),
        "positive_rate": float(y_pred_prob.mean()),
        "auc_score": float(roc_auc_score(y_true, y_pred_prob)),
        "logloss": float(log_loss(y_true, y_pred_prob))
    }

# MAIN

def main(snapshot_date: str, model_name: str):
    print("📊 Starting model monitoring…")

    # Load inference output
    csv_path = f"scripts/model_outputs/inference_{snapshot_date}.csv"
    df = pd.read_csv(csv_path)

    # Compute metrics
    metrics = compute_metrics(df)
    metrics["model_name"] = model_name
    metrics["snapshot_date"] = snapshot_date

    # Insert into PostgreSQL
    conn = connect_with_retry()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO model_metrics (model_name, snapshot_date, inference_count, positive_rate, auc_score, logloss)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        metrics["model_name"],
        metrics["snapshot_date"],
        metrics["inference_count"],
        metrics["positive_rate"],
        metrics["auc_score"],
        metrics["logloss"]
    ))
    conn.commit()
    cur.close()
    conn.close()

    print("✅ Monitoring metrics saved to PostgreSQL")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    parser.add_argument("--modelname", required=True)
    args = parser.parse_args()

    main(args.snapshotdate, args.modelname)
