import logging
import re
from pathlib import Path
from typing import List, Optional
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler


def load_category_by_date(base_path, data_window):
    path = Path(base_path)
    pattern = re.compile(r".*_(\d{4}-\d{2}-\d{2})")
    folders = [f for f in path.iterdir() if f.is_dir() and pattern.match(f.name)]
    if data_window:
        date_strs = [d if isinstance(d, str) else d.strftime("%Y-%m-%d") for d in data_window]
        folders = [f for f in folders if pattern.match(f.name).group(1) in date_strs]
    if not folders:
        return None
    return spark.read.parquet(*[str(f) for f in folders])

def create_feature_store(
    silver_root: str,
    output_path: str,
    data_window: Optional[List[str]] = None,
    categorical_cols=None,
    numeric_cols=None
):
    spark = SparkSession.builder.appName("FeatureStoreCreation").getOrCreate()

    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        filename="logs/gold_feature_store.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # === Dates ===
    if data_window:
        date_list = [d.strftime("%Y-%m-%d") if not isinstance(d, str) else d for d in data_window]
    
    # === Load Each Feature Category ===
    logging.info("Loading silver-level parquet folders")
    loan_df        = load_category_by_date(f"{silver_root}/loan_terms", date_list)
    demo_df        = load_category_by_date(f"{silver_root}/demographic", date_list)
    fin_df         = load_category_by_date(f"{silver_root}/financial", date_list)
    credit_df      = load_category_by_date(f"{silver_root}/credit_history", date_list)

    if not all([loan_df, demo_df, fin_df, credit_df]):
        logging.warning("One or more silver categories could not be loaded.")
        return

    # === Join all on member_id and snapshot_date ===
    logging.info("Joining all dataframes on member_id and snapshot_date")
    df = loan_df.join(demo_df, ["member_id", "snapshot_date"], "outer") \
                .join(fin_df, ["member_id", "snapshot_date"], "outer") \
                .join(credit_df, ["member_id", "snapshot_date"], "outer")

    df = df.withColumn("snapshot_date", to_date("snapshot_date"))

    # === Select Features ===
    