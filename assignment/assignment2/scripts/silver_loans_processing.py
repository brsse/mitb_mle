import os
import argparse
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_silver_loans(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # Load bronze data from Parquet partition
    partition_path = os.path.join(bronze_lms_directory, "loan_daily", f"snapshot_date={snapshot_date_str}")
    df = spark.read.parquet(partition_path)
    print(f"✅ Loaded from: {partition_path} | Row count: {df.count()}")

    column_type_map = {
        "loan_id": StringType(), "Customer_ID": StringType(), "loan_start_date": DateType(),
        "tenure": IntegerType(), "installment_num": IntegerType(), "loan_amt": FloatType(),
        "due_amt": FloatType(), "paid_amt": FloatType(), "overdue_amt": FloatType(),
        "balance": FloatType(), "snapshot_date": DateType()
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", 
            F.when(col("installments_missed") > 0, 
                   F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", 
            F.when(col("overdue_amt") > 0.0, 
                   F.datediff(col("snapshot_date"), col("first_missed_date")))
            .otherwise(0).cast(IntegerType()))

    # Save as partitioned Parquet
    df.write.partitionBy("snapshot_date").mode("overwrite") \
        .parquet(os.path.join(silver_loan_daily_directory, "loans_clean"))
    print(f"💾 Saved silver → loans_clean")

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    parser.add_argument("--bronzedir", required=True)
    parser.add_argument("--silverdir", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("silver_loan_daily").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_silver_loans(args.snapshotdate, args.bronzedir, args.silverdir, spark)
    spark.stop()

if __name__ == "__main__":
    main()
