import psycopg2
from psycopg2 import OperationalError
import time

def connect_with_retry(retries=5, delay=5):
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(
                dbname="airflow",
                user="airflow",
                password="airflow",
                host="postgres",
                port="5432"
            )
            print("✅ Connected to PostgreSQL")
            return conn
        except OperationalError as e:
            print(f"⏳ Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    raise Exception("❌ Could not connect to PostgreSQL after multiple retries.")

# Run setup
if __name__ == "__main__":
    conn = connect_with_retry(retries=10, delay=5)  # You can tune these values
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS model_metrics (
        id SERIAL PRIMARY KEY,
        model_name TEXT,
        snapshot_date DATE,
        inference_count INT,
        positive_rate FLOAT,
        auc_score FLOAT,
        logloss FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    cur.close()
    conn.close()

    print("✅ Table created")