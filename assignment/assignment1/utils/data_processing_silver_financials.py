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

from pyspark.sql.functions import col, regexp_extract, regexp_replace, when, coalesce, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": FloatType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),     
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }

    # Cast columns
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # clean data
    df = df.withColumn("Amount_invested_monthly", regexp_replace(col("Amount_invested_monthly"), "[^\\d.]", "").cast("float"))
    df = df.withColumn("Payment_Behaviour", when(col("Payment_Behaviour") == "!@9#%8", None).otherwise(col("Payment_Behaviour")))
    df = df.withColumn("Changed_Credit_Limit", regexp_replace(col("Changed_Credit_Limit"), "[^\\d.]", "").cast("float"))
    df = df.withColumn("Num_of_Delayed_Payment", regexp_replace(col("Num_of_Delayed_Payment"), "[^\\d]", "").cast("int"))
    df = df.withColumn("Annual_Income", regexp_replace(col("Annual_Income"), "[^\\d]", "").cast("float"))
    df = df.withColumn("Num_of_Loan", regexp_replace(col("Num_of_Loan"), "[^\\d]", "").cast("int"))
    df = df.withColumn("Num_of_Loan_Overcap_Flag", when(col("Num_of_Loan") > 20, 1).otherwise(0))
    df = df.withColumn("Num_of_Loan", when(col("Num_of_Loan") > 20, 20).otherwise(col("Num_of_Loan")))
    
    # encode categorical features
    df = df.withColumn("Payment_of_Min_Amount", when(col("Payment_of_Min_Amount") == "NM", None)
                       .when(col("Payment_of_Min_Amount") == "No", 0)
                       .when(col("Payment_of_Min_Amount") == "Yes", 1)
                       .otherwise(None).cast(IntegerType())
                      )
    df = df.withColumn("Credit_Mix", when(col("Credit_Mix") == "_", None).otherwise(col("Credit_Mix")))
    df = df.withColumn("Credit_Mix", when(col("Credit_Mix") == "Bad", 0)
                       .when(col("Credit_Mix") == "Standard", 1)
                       .when(col("Credit_Mix") == "Good", 2)
                       .otherwise(None).cast(IntegerType())
                      )

    # extracting numeric data from credit_history_age
    df = df.withColumn("credit_history_yrs", regexp_extract("Credit_History_Age", r"(\\d+)\\s+Years?", 1).cast(IntegerType()))
    df = df.withColumn("credit_history_mths", regexp_extract("Credit_History_Age", r"(\\d+)\\s+Months?", 1).cast(IntegerType()))
    df = df.withColumn("Credit_History_in_Months", 
                       (coalesce(col("credit_history_yrs"), lit(0)) * 12 + 
                        coalesce(col("credit_history_mths"), lit(0)))
                       .cast(IntegerType()))

    # augment data: debt to income ratio
    df = df.withColumn("Debt_to_income", col("Outstanding_Debt") / (col("Annual_Income") + F.lit(1)))

    # augment data: loan to income ratio
    df = df.withColumn("Loan_to_income", col("Total_EMI_per_month") / (col("Monthly_Inhand_Salary") + F.lit(1)))
   
    # augment data: monthly income investment rate
    df = df.withColumn("Investment_rate", col("Amount_invested_monthly") / (col("Monthly_Inhand_Salary") + F.lit(1)))

    # augment data: payments missed to total loans ratio
    df = df.withColumn("Delayed_payment_ratio", col("Num_of_Delayed_Payment") / (col("Num_of_Loan") + F.lit(1)))

    # drop unnecessary columns
    df = df.drop("Credit_History_Age", "Changed_Credit_Limit",)
    
    # save silver table - IRL connect to database to write
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df