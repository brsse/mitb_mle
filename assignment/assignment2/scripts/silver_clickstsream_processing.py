import os
import argparse
from datetime import datetime
import pyspark
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, DateType

def process_silver_clickstream(snapshot_date_str, bronze_dir, silver_dir, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Load bronze data from Parquet partition
    partition_path = os.path.join(bronze_dir, "clickstream", f"snapshot_date={snapshot_date_str}")
    df = spark.read.parquet(partition_path)
    print(f"✅ Loaded: {partition_path} → {df.count()} rows")

    # Define schema explicitly
    column_type_map = {
        f"fe_{i}": IntegerType() for i in range(1, 21)
    }
    column_type_map.update({
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    })

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Save as partitioned Parquet
    output_path = os.path.join(silver_dir, "clickstream_clean")
    df.write.partitionBy("snapshot_date").mode("overwrite").parquet(output_path)
    print(f"💾 Saved silver → clickstream_clean")

    return df

# --- Entry point for Airflow CLI call ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("silver_clickstream").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    bronze_dir = "/opt/airflow/scripts/datamart/bronze"
    silver_dir = "/opt/airflow/scripts/datamart/silver"
    process_silver_clickstream(args.snapshotdate, bronze_dir, silver_dir, spark)

    spark.stop()
