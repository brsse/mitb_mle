import argparse
import os
import pickle
import pyspark
import pprint
import psycopg2
from datetime import datetime, timedelta

import pandas as pd
from pyspark.sql.functions import col, lit, min as F_min, max as F_max

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import xgboost as xgb

def load_parquet_as_pandas(spark, path, start_date, end_date):
    print(f"Loading data from: {path}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Read from partitioned Parquet and filter by date range
    sdf = spark.read.parquet(path)
    print(f"Total records before filtering: {sdf.count()}")
    
    # Filter by date range
    sdf = sdf.withColumn("snapshot_date", col("snapshot_date").cast("date"))
    sdf = sdf.filter((col("snapshot_date") >= lit(start_date)) & (col("snapshot_date") < lit(end_date)))
    
    filtered_count = sdf.count()
    print(f"Records after date filtering: {filtered_count}")
    
    if filtered_count == 0:
        # If no data in the exact range, try to get whatever data is available
        print("⚠️  No data found for exact date range. Checking for available data...")
        all_data = spark.read.parquet(path)
        all_data = all_data.withColumn("snapshot_date", col("snapshot_date").cast("date"))
        
        # Get the date range of available data
        min_date = all_data.agg(F_min("snapshot_date")).collect()[0][0]
        max_date = all_data.agg(F_max("snapshot_date")).collect()[0][0]
        
        print(f"Available data range: {min_date} to {max_date}")
        
        if min_date is None or max_date is None:
            raise ValueError(f"No data available in {path}")
        
        # Use whatever data is available within the requested range
        sdf = all_data.filter((col("snapshot_date") >= lit(max(min_date, start_date))) & 
                             (col("snapshot_date") <= lit(min(max_date, end_date - timedelta(days=1)))))
        
        final_count = sdf.count()
        print(f"Using available data: {final_count} records")
        
        if final_count == 0:
            raise ValueError(f"No data available in the requested range or nearby dates")
    
    return sdf.toPandas()

def save_metrics_to_postgres(model_name, auc_score, f1_score, accuracy, training_date, start_date, end_date):
    """Save model metrics to PostgreSQL database"""
    try:
        # PostgreSQL connection parameters
        conn_params = {
            'host': 'postgres',  # Docker service name
            'database': 'airflow',
            'user': 'airflow',
            'password': 'airflow',
            'port': 5432
        }
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        # Create metrics table if it doesn't exist
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            auc_score DECIMAL(5,4),
            f1_score DECIMAL(5,4),
            accuracy DECIMAL(5,4),
            training_date DATE NOT NULL,
            training_start_date DATE NOT NULL,
            training_end_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_sql)
        
        # Insert metrics
        insert_sql = """
        INSERT INTO model_metrics 
        (model_name, auc_score, f1_score, accuracy, training_date, training_start_date, training_end_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_sql, (
            model_name, auc_score, f1_score, accuracy, 
            training_date, start_date, end_date
        ))
        
        # Commit and close
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Metrics saved to PostgreSQL for model: {model_name}")
        
    except Exception as e:
        print(f"❌ Error saving metrics to PostgreSQL: {e}")
        # Don't fail the training if metrics saving fails

def main(start_date_str, end_date_str, model_name):
    print("\n--- Starting model training and inference ---\n")
    
    # Setup
    spark = pyspark.sql.SparkSession.builder.appName("ModelPipeline").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    config = {
        "start_date": start_date,
        "end_date": end_date,
        "model_name": model_name,
        "model_bank_dir": "/opt/airflow/scripts/model_bank/",
        "feature_path": "/opt/airflow/scripts/datamart/gold/feature_store/",
        "label_path": "/opt/airflow/scripts/datamart/gold/label_store/",
    }
    pprint.pprint(config)

    # Load data
    features_pdf = load_parquet_as_pandas(spark, config["feature_path"], start_date, end_date)
    labels_pdf = load_parquet_as_pandas(spark, config["label_path"], start_date, end_date)

    if features_pdf.empty:
        raise ValueError("Features DataFrame is empty. Check feature store path or snapshot range.")
    if labels_pdf.empty:
        raise ValueError("Labels DataFrame is empty. Check label store path or snapshot range.")

    df = pd.merge(features_pdf, labels_pdf, on=["Customer_ID", "loan_start_date"])
    if df.empty:
        raise ValueError("No matching records after merging features and labels.")
    feature_cols = [col for col in df.columns if col.startswith("fe_")]
    
    X = df[feature_cols]
    y = df["label"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    auc_score = roc_auc_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    print("AUC on test set:", auc_score)
    print("F1-Score on test set:", f1)
    print("Accuracy on test set:", acc)

    # Save model
    os.makedirs(config["model_bank_dir"], exist_ok=True)
    model_artifact = {
        "model": model,
        "preprocessing_transformers": {"stdscaler": scaler}
    }
    model_path = os.path.join(config["model_bank_dir"], f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_artifact, f)
    print(f"Model saved at: {model_path}")

    # Save metrics to PostgreSQL
    save_metrics_to_postgres(model_name, auc_score, f1, acc, datetime.now().date(), start_date, end_date)

    spark.stop()
    print("\n--- Job complete ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--startdate", required=True, help="Training window start YYYY-MM-DD")
    parser.add_argument("--enddate", required=True, help="Training window end YYYY-MM-DD")
    parser.add_argument("--modelname", required=True)
    args = parser.parse_args()
    main(args.startdate, args.enddate, args.modelname)