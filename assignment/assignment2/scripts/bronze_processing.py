import os
import argparse
from datetime import datetime
import pyspark
from pyspark.sql.functions import col

data_dir = '/opt/airflow/scripts/data'

def main(snapshotdate):
    spark = pyspark.sql.SparkSession.builder.appName("bronze_processing").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    bronze_output_directory = "/opt/airflow/scripts/datamart/bronze/"
    process_bronze_table(snapshotdate, bronze_output_directory, spark)

    spark.stop()

def process_bronze_table(snapshot_date_str, bronze_output_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    sources = {
        "loan_daily": f"{data_dir}/lms_loan_daily.csv",
        "clickstream": f"{data_dir}/feature_clickstream.csv",
        "attributes": f"{data_dir}/features_attributes.csv",
        "financials": f"{data_dir}/features_financials.csv"
    }

    for name, path in sources.items():
        df = spark.read.csv(path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
        print(f"{snapshot_date_str} row count for {name}:", df.count())

        output_path = os.path.join(bronze_output_directory, name, f"snapshot_date={snapshot_date_str}")
        df.write.mode("overwrite").parquet(output_path)
        print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    main(args.snapshotdate)