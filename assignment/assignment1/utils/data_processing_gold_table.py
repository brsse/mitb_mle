import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col, when, isnan, split, array_contains
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_features_gold_table(snapshot_date_str, silver_directory, gold_feature_store_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    print("Processing Gold feature table for", snapshot_date)

    # connect to silver tables
    attr_path = os.path.join(silver_directory, "attributes", f"silver_attributes_{snapshot_date_str.replace('-','_')}.parquet")
    click_path = os.path.join(silver_directory, "clickstream", f"silver_clickstream_{snapshot_date_str.replace('-','_')}.parquet")
    fin_path = os.path.join(silver_directory, "financials", f"silver_financials_{snapshot_date_str.replace('-','_')}.parquet")
    loan_path = os.path.join(silver_directory, "loan_daily", f"silver_loan_daily_{snapshot_date_str.replace('-','_')}.parquet")

    attr_df = spark.read.parquet(attr_path).dropDuplicates(["Customer_ID", "snapshot_date"])
    click_df = spark.read.parquet(click_path)
    fin_df = spark.read.parquet(fin_path).dropDuplicates(["Customer_ID", "snapshot_date"])
    loan_df = spark.read.parquet(loan_path)
    print(f"Files loaded from: {attr_df} with {attr_df.count()} rows\n\
                            {click_df} with {click_df.count()} rows\n\
                            {fin_df} with {fin_df.count()} rows\n\
                            {loan_df} with {loan_df.count()} rows") 

    # add filter for age
    if "Age" in attr_df.columns:
        attr_df = attr_df.withColumn("Has_Valid_Age", 
                           when(col("Age").isNotNull() & ~isnan(col("Age")), 1).otherwise(0)
                          )
    
    # one-hot encode columns
    if "Occupation" in attr_df.columns:
        unique_occupations = attr_df.select("Occupation").distinct().rdd.flatMap(lambda x: x).collect()
        for occ in unique_occupations:
            if occ is not None:
                col_suffix = str(occ).strip().replace(" ", "_").replace("-", "_").lower()
                attr_df = attr_df.withColumn(f"Occupation_{col_suffix}", when(col("Occupation") == occ, 1).otherwise(0))
        attr_df = attr_df.drop("Occupation")

    if "Payment_Behaviour" in fin_df.columns:
        unique_behaviours = fin_df.select("Payment_Behaviour").distinct().rdd.flatMap(lambda x: x).collect()
        for val in unique_behaviours:
            if val is not None:
                suffix = val.strip().replace(" ", "_").replace("-", "_").lower()
                fin_df = fin_df.withColumn(f"payment_behaviour_{suffix}", when(col("Payment_Behaviour") == val, 1).otherwise(0))
        fin_df = fin_df.drop("Payment_Behaviour")

    # multi-hot encode type_of_loan
    if "Type_of_Loan" in fin_df.columns:
        fin_df = fin_df.withColumn("loan_types_array", split(col("Type_of_Loan"), ",\\s*"))
        loan_type_rows = fin_df.select("loan_types_array").where(col("loan_types_array").isNotNull())
        unique_loan_types = (loan_type_rows.rdd
                             .flatMap(lambda row: row.loan_types_array if row.loan_types_array is not None else [])
                             .distinct().collect())
        for loan_type in unique_loan_types:
            if loan_type is not None:
                col_suffix = loan_type.strip().replace(" ", "_").replace("-", "_").lower()
                fin_df = fin_df.withColumn(f"loan_type_{col_suffix}", 
                                           when(array_contains(col("loan_types_array"), loan_type), 1).otherwise(0))
        fin_df = fin_df.drop("Type_of_Loan", "loan_types_array")
    
    # join all features on customer_id
    df = loan_df.join(attr_df, on=["Customer_ID", "snapshot_date"], how="left") \
                .join(fin_df, on=["Customer_ID", "snapshot_date"], how="left") \
                .join(click_df, on=["Customer_ID", "snapshot_date"], how="left")

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df