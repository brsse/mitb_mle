import os
from datetime import datetime
import argparse
import pyspark
from pyspark.sql.functions import col, when, regexp_replace
from pyspark.sql.types import StringType, IntegerType, DateType

def process_silver_attributes(snapshot_date_str, bronze_dir, silver_dir, spark):
    
    # Load bronze data from Parquet partition
    partition_path = os.path.join(bronze_dir, "attributes", f"snapshot_date={snapshot_date_str}")
    df = spark.read.parquet(partition_path)
    print(f"✅ Loaded: {partition_path} → {df.count()} rows")

    # Enforce schema
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Clean age and occupation
    df = df.withColumn("Age_clean", regexp_replace(col("Age"), "[^\\d]", "").cast("int")) \
           .filter((col("Age_clean") >= 18) & (col("Age_clean") <= 100)) \
           .drop("Age").withColumnRenamed("Age_clean", "Age") \
           .withColumn("Occupation", when(col("Occupation") == "_______", None).otherwise(col("Occupation"))) \
           .fillna({"Occupation": "Unknown"})

    # Drop PII
    df = df.drop("Name", "SSN")

    # Save as partitioned Parquet
    df.write.partitionBy("snapshot_date").mode("overwrite") \
        .parquet(os.path.join(silver_dir, "attributes_clean"))
    print(f"💾 Saved silver → attributes_clean")

    return df

# --- Entry point for Airflow CLI call ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("silver_attributes").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    bronze_dir = "/opt/airflow/scripts/datamart/bronze"
    silver_dir = "/opt/airflow/scripts/datamart/silver"
    process_silver_attributes(args.snapshotdate, bronze_dir, silver_dir, spark)

    spark.stop()
