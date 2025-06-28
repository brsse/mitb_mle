import os
import argparse
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, regexp_extract, regexp_replace, when, coalesce, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_financials(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    # Load bronze data from Parquet partition
    partition_path = os.path.join(bronze_financials_directory, "financials", f"snapshot_date={snapshot_date_str}")
    df = spark.read.parquet(partition_path)
    print(f"✅ Loaded from: {partition_path} | Row count: {df.count()}")

    column_type_map = {
        "Customer_ID": StringType(), "Annual_Income": FloatType(), "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(), "Num_Credit_Card": IntegerType(), "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(), "Type_of_Loan": StringType(), "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(), "Changed_Credit_Limit": FloatType(), "Num_Credit_Inquiries": FloatType(),
        "Credit_Mix": StringType(), "Outstanding_Debt": FloatType(), "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(), "Payment_of_Min_Amount": StringType(),     
        "Total_EMI_per_month": FloatType(), "Amount_invested_monthly": FloatType(), 
        "Payment_Behaviour": StringType(), "Monthly_Balance": FloatType(), "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    df = df.withColumn("Amount_invested_monthly", regexp_replace(col("Amount_invested_monthly"), "[^\\d.]", "").cast("float")) \
           .withColumn("Payment_Behaviour", when(col("Payment_Behaviour") == "!@9#%8", None).otherwise(col("Payment_Behaviour"))) \
           .withColumn("Changed_Credit_Limit", regexp_replace(col("Changed_Credit_Limit"), "[^\\d.]", "").cast("float")) \
           .withColumn("Num_of_Delayed_Payment", regexp_replace(col("Num_of_Delayed_Payment"), "[^\\d]", "").cast("int")) \
           .withColumn("Annual_Income", regexp_replace(col("Annual_Income"), "[^\\d]", "").cast("float")) \
           .withColumn("Num_of_Loan", regexp_replace(col("Num_of_Loan"), "[^\\d]", "").cast("int")) \
           .withColumn("Num_of_Loan_Overcap_Flag", when(col("Num_of_Loan") > 20, 1).otherwise(0)) \
           .withColumn("Num_of_Loan", when(col("Num_of_Loan") > 20, 20).otherwise(col("Num_of_Loan")))

    df = df.withColumn("Payment_of_Min_Amount", 
                       when(col("Payment_of_Min_Amount") == "NM", None)
                       .when(col("Payment_of_Min_Amount") == "No", 0)
                       .when(col("Payment_of_Min_Amount") == "Yes", 1)
                       .otherwise(None).cast(IntegerType()))

    df = df.withColumn("Credit_Mix", 
                       when(col("Credit_Mix") == "_", None)
                       .when(col("Credit_Mix") == "Bad", 0)
                       .when(col("Credit_Mix") == "Standard", 1)
                       .when(col("Credit_Mix") == "Good", 2)
                       .otherwise(None).cast(IntegerType()))

    df = df.withColumn("credit_history_yrs", regexp_extract("Credit_History_Age", r"(\d+)\s+Years?", 1).cast(IntegerType())) \
           .withColumn("credit_history_mths", regexp_extract("Credit_History_Age", r"(\d+)\s+Months?", 1).cast(IntegerType())) \
           .withColumn("Credit_History_in_Months", 
                       (coalesce(col("credit_history_yrs"), lit(0)) * 12 + 
                        coalesce(col("credit_history_mths"), lit(0)))
                       .cast(IntegerType()))

    df = df.withColumn("Debt_to_income", col("Outstanding_Debt") / (col("Annual_Income") + F.lit(1))) \
           .withColumn("Loan_to_income", col("Total_EMI_per_month") / (col("Monthly_Inhand_Salary") + F.lit(1))) \
           .withColumn("Investment_rate", col("Amount_invested_monthly") / (col("Monthly_Inhand_Salary") + F.lit(1))) \
           .withColumn("Delayed_payment_ratio", col("Num_of_Delayed_Payment") / (col("Num_of_Loan") + F.lit(1)))

    df = df.drop("Credit_History_Age", "Changed_Credit_Limit")

    # Save as partitioned Parquet
    df.write.partitionBy("snapshot_date").mode("overwrite") \
        .parquet(os.path.join(silver_financials_directory, "financials_clean"))
    print(f"💾 Saved silver → financials_clean")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    parser.add_argument("--bronzedir", required=True)
    parser.add_argument("--silverdir", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("silver_financials").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_silver_financials(args.snapshotdate, args.bronzedir, args.silverdir, spark)
    spark.stop()


if __name__ == "__main__":
    main()
