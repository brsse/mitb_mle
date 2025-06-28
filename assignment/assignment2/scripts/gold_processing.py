import os
from datetime import datetime
import argparse
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, isnan, split, array_contains, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_features_gold_table(snapshot_date_str, silver_directory, gold_feature_store_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    print("Processing Gold feature table for", snapshot_date)

    # Read from partitioned silver data
    attr_path = os.path.join(silver_directory, "attributes_clean")
    click_path = os.path.join(silver_directory, "clickstream_clean")
    fin_path = os.path.join(silver_directory, "financials_clean")
    loan_path = os.path.join(silver_directory, "loans_clean")

    # Filter by snapshot date
    attr_df = spark.read.parquet(attr_path).filter(col("snapshot_date") == lit(snapshot_date)).dropDuplicates(["Customer_ID", "snapshot_date"])
    click_df = spark.read.parquet(click_path).filter(col("snapshot_date") == lit(snapshot_date))
    fin_df = spark.read.parquet(fin_path).filter(col("snapshot_date") == lit(snapshot_date)).dropDuplicates(["Customer_ID", "snapshot_date"])
    loan_df = spark.read.parquet(loan_path).filter(col("snapshot_date") == lit(snapshot_date))

    # one-hot encode occupation
    if "Occupation" in attr_df.columns:
        for occ in attr_df.select("Occupation").distinct().rdd.flatMap(lambda x: x).collect():
            if occ:
                suffix = occ.strip().replace(" ", "_").replace("-", "_").lower()
                attr_df = attr_df.withColumn(f"Occupation_{suffix}", when(col("Occupation") == occ, 1).otherwise(0))
        attr_df = attr_df.drop("Occupation")

    # one-hot encode payment behaviour
    if "Payment_Behaviour" in fin_df.columns:
        for pb in fin_df.select("Payment_Behaviour").distinct().rdd.flatMap(lambda x: x).collect():
            if pb:
                suffix = pb.strip().replace(" ", "_").replace("-", "_").lower()
                fin_df = fin_df.withColumn(f"payment_behaviour_{suffix}", when(col("Payment_Behaviour") == pb, 1).otherwise(0))
        fin_df = fin_df.drop("Payment_Behaviour")

    # multi-hot encode loan types
    if "Type_of_Loan" in fin_df.columns:
        fin_df = fin_df.withColumn("loan_types_array", split(col("Type_of_Loan"), ",\\s*"))
        types = fin_df.select("loan_types_array").rdd.flatMap(lambda r: r.loan_types_array or []).distinct().collect()
        for t in types:
            if t:
                suffix = t.strip().replace(" ", "_").replace("-", "_").lower()
                fin_df = fin_df.withColumn(f"loan_type_{suffix}", when(array_contains(col("loan_types_array"), t), 1).otherwise(0))
        fin_df = fin_df.drop("Type_of_Loan", "loan_types_array")

    # join all
    df = loan_df.join(attr_df,  ["Customer_ID", "snapshot_date"], "left") \
                .join(fin_df,   ["Customer_ID", "snapshot_date"], "left") \
                .join(click_df, ["Customer_ID", "snapshot_date"], "left")

    # Save as partitioned Parquet
    df.write.partitionBy("snapshot_date").mode("overwrite") \
        .parquet(gold_feature_store_directory)
    print("✅ Feature store saved to:", gold_feature_store_directory)
    return df


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd=30, mob=6):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Read from partitioned silver data
    df = spark.read.parquet(silver_loan_daily_directory).filter(col("snapshot_date") == lit(snapshot_date))
    print(f"Loaded loans for label from: {silver_loan_daily_directory} with {df.count()} rows")

    df = df.filter(col("mob") == mob)
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", lit(f"{dpd}dpd_{mob}mob"))

    output = df.select("loan_id", "Customer_ID", "loan_start_date", "label", "label_def", "snapshot_date")
    
    # Save as partitioned Parquet
    output.write.partitionBy("snapshot_date").mode("overwrite") \
        .parquet(gold_label_store_directory)
    print("✅ Label store saved to:", gold_label_store_directory)
    return output

def main(snapshot_date, task):
    spark = pyspark.sql.SparkSession.builder.appName("gold_processing").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    silver_dir = "/opt/airflow/scripts/datamart/silver/"
    gold_feature_dir = "/opt/airflow/scripts/datamart/gold/feature_store/"
    gold_label_dir = "/opt/airflow/scripts/datamart/gold/label_store/"
    silver_loans_dir = os.path.join(silver_dir, "loans_clean")

    if task == "features":
        process_features_gold_table(snapshot_date, silver_dir, gold_feature_dir, spark)
    elif task == "labels":
        process_labels_gold_table(snapshot_date, silver_loans_dir, gold_label_dir, spark, dpd=30, mob=6)
    else:
        raise ValueError(f"Unknown task: {task}")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    parser.add_argument("--task", required=True, choices=["features", "labels"])
    args = parser.parse_args()
    main(args.snapshotdate, args.task)
