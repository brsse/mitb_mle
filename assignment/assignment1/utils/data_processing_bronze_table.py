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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table(snapshot_date_str, bronze_output_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # source directories
    sources = {
        "loan_daily": "data/lms_loan_daily.csv",
        "clickstream": "data/feature_clickstream.csv",
        "attributes": "data/features_attributes.csv",
        "financials": "data/features_financials.csv"
    }

    for name, path in sources.items():
        # load data
        df = spark.read.csv(path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
        print(snapshot_date_str + 'row count:', df.count())

        # save bronze table to datamart
        partition_name = f"bronze_{name}_" + snapshot_date_str.replace('-','_') + '.csv'
        
        dataset_folder = f"{bronze_output_directory}{name}/"
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        filepath = dataset_folder + partition_name
        df.toPandas().to_csv(filepath, index=False)
        print('saved to:', filepath)
    
    # # load data - IRL ingest from back end source system
    # df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    # print(snapshot_date_str + 'row count:', df.count())

    # # save bronze table to datamart - IRL connect to database to write
    # partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    # filepath = bronze_output_directory + partition_name
    # df.toPandas().to_csv(filepath, index=False)
    # print('saved to:', filepath)

    return df
