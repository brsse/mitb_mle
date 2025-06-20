import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col, count, when, isnan
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, NumericType

import utils.data_processing_bronze_table
import utils.data_processing_silver_loan_daily
import utils.data_processing_silver_attributes
import utils.data_processing_silver_clickstream
import utils.data_processing_silver_financials
import utils.data_processing_gold_table

# set up pyspark session
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    first_of_month_dates = []

    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
dates_str_lst

# create bronze datalake
bronze_output_directory = "datamart/bronze/"

if not os.path.exists(bronze_output_directory):
    os.makedirs(bronze_output_directory)

# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_output_directory, spark)

# create silver datalake
bronze_lms_directory = "datamart/bronze/loan_daily/"
bronze_attributes_directory = "datamart/bronze/attributes/"
bronze_clickstream_directory = "datamart/bronze/clickstream/"
bronze_financials_directory = "datamart/bronze/financials/"

silver_loan_daily_directory = "datamart/silver/loan_daily/"
silver_attributes_directory = "datamart/silver/attributes/"
silver_clickstream_directory = "datamart/silver/clickstream/"
silver_financials_directory = "datamart/silver/financials/"

if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)
if not os.path.exists(silver_attributes_directory):
    os.makedirs(silver_attributes_directory)
if not os.path.exists(silver_clickstream_directory):
    os.makedirs(silver_clickstream_directory)
if not os.path.exists(silver_financials_directory):
    os.makedirs(silver_financials_directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_loan_daily.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
    utils.data_processing_silver_attributes.process_silver_table(date_str, bronze_attributes_directory, silver_attributes_directory, spark)
    utils.data_processing_silver_clickstream.process_silver_table(date_str, bronze_clickstream_directory, silver_clickstream_directory, spark)
    utils.data_processing_silver_financials.process_silver_table(date_str, bronze_financials_directory, silver_financials_directory, spark)

# create gold datalake
gold_feature_store_directory = "datamart/gold/feature_store/"

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

# run gold backfill
silver_base_directory = "datamart/silver/"
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_features_gold_table(date_str, silver_base_directory, gold_feature_store_directory, spark)

# inspect gold level feature store
folder_path = gold_feature_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()
