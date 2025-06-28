import os
import argparse
from datetime import datetime
import pyspark

# Import processing functions from your existing scripts
from silver_attributes_processing import process_silver_attributes as process_attributes
from silver_clickstsream_processing import process_silver_clickstream as process_clickstream
from silver_financials_processing import process_silver_financials as process_financials
from silver_loans_processing import process_silver_loans as process_loan_daily

def main(snapshot_date: str):
    spark = pyspark.sql.SparkSession.builder.appName("silver_processing").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    bronze_dir = "/opt/airflow/scripts/datamart/bronze/"
    silver_dir = "/opt/airflow/scripts/datamart/silver/"

    print(f"📅 Processing snapshot date: {snapshot_date}")

    # Run each processing module
    print("🔁 Processing attributes")
    process_attributes(snapshot_date, bronze_dir, silver_dir, spark)

    print("🔁 Processing clickstream")
    process_clickstream(snapshot_date, bronze_dir, silver_dir, spark)

    print("🔁 Processing financials")
    process_financials(snapshot_date, bronze_dir, silver_dir, spark)

    print("🔁 Processing loan_daily")
    process_loan_daily(snapshot_date, bronze_dir, silver_dir, spark)

    spark.stop()
    print("✅ All silver tables processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True, help="Snapshot date in YYYY-MM-DD format")
    args = parser.parse_args()
    main(args.snapshotdate)
