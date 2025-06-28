#!/usr/bin/env python3
"""
Script to view model metrics from PostgreSQL database
"""

import psycopg2
import pandas as pd
from datetime import datetime

def get_model_metrics():
    """Retrieve model metrics from PostgreSQL"""
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
        
        # Query metrics
        query = """
        SELECT 
            model_name,
            auc_score,
            f1_score,
            accuracy,
            training_date,
            training_start_date,
            training_end_date,
            created_at
        FROM model_metrics 
        ORDER BY created_at DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
        
    except Exception as e:
        print(f"❌ Error retrieving metrics from PostgreSQL: {e}")
        return None

def main():
    print("📊 Model Metrics from PostgreSQL\n")
    
    df = get_model_metrics()
    
    if df is not None and not df.empty:
        print(f"Found {len(df)} model training records:\n")
        print(df.to_string(index=False))
        
        # Summary statistics
        print(f"\n📈 Summary Statistics:")
        print(f"Average AUC: {df['auc_score'].mean():.4f}")
        print(f"Average F1-Score: {df['f1_score'].mean():.4f}")
        print(f"Average Accuracy: {df['accuracy'].mean():.4f}")
        
        # Latest model
        latest = df.iloc[0]
        print(f"\n🏆 Latest Model ({latest['model_name']}):")
        print(f"  AUC: {latest['auc_score']:.4f}")
        print(f"  F1-Score: {latest['f1_score']:.4f}")
        print(f"  Accuracy: {latest['accuracy']:.4f}")
        print(f"  Trained on: {latest['training_date']}")
        print(f"  Data period: {latest['training_start_date']} to {latest['training_end_date']}")
        
    else:
        print("No metrics found in the database.")

if __name__ == "__main__":
    main() 