import logging
import re
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, weekofyear, month, date_sub, dayofweek, lit, concat_ws


def create_gold_label_store(silver_dir, gold_dir, data_window=None):
    spark = SparkSession.builder.appName("GoldLabelStore").getOrCreate()

    # === Logging ===
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename="logs/gold_label_store.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info(f"Creating Gold Label Store from {silver_dir}...")

    silver_dir = Path(silver_dir)
    pattern = re.compile(r"loan_terms_(\d{4}-\d{2}-\d{2})")
    
    all_folders = [f for f in silver_dir.iterdir() if f.is_dir() and pattern.match(f.name)]
    for f in silver_dir.iterdir():
        logging.info(f"Checking folder: {f.name}")
        if pattern.match(f.name):
            logging.info(f"Match: {f.name}")

    # === Filter By Window ===
    date_list = [d.strftime("%Y-%m-%d") for d in data_window]
    if data_window:
        logging.info(f"Filtering for snapshot_date in: {data_window}")
        matched_folders = [
            str(f) for f in all_folders if pattern.match(f.name).group(1) in date_list
        ]
    else:
        logging.info("No date filter provided; reading all folders")
        matched_folders = [str(f) for f in all_folders]

    if not matched_folders:
        logging.warning("No matching folders found. Exiting.")
        print("No matching folders found. Exiting.")
        return

    logging.info(f"Processing data from {matched_folders}")
    df = spark.read.parquet(*matched_folders)

    selected_columns = ["member_id", "snapshot_date", "grade"]
    df = df.select(*selected_columns).filter("grade IS NOT NULL")

    # === Convert snapshot_date For Partitioning ===
    logging.info(f"Creating date, week and month columns using snapshot_date")
    df = df.withColumn("snapshot_date", to_date("snapshot_date"))
    df = df.withColumn("start_of_week", date_sub("snapshot_date", dayofweek("snapshot_date") - 2))
    df = df.withColumn("snapshot_week", weekofyear("snapshot_date"))
    df = df.withColumn("snapshot_month", month("snapshot_date"))

    # === Custom Parquet Folder Names ===
    # df = df.withColumn(
    #     "label_store_partition",
    #     concat_ws("_", lit("label_store_"), df["snapshot_date_str"])   # day-based partition
    # )
    df = df.withColumn(
        "label_store_partition",
        concat_ws("_", lit("label_store_week"), df["snapshot_week"], df["start_of_week"])   # week-based partition
    )
    # df = df.withColumn(
    #     "label_store_partition",
    #     concat_ws("_", lit("label_store_month"), df["snapshot_month"])   # month-based partition
    # )
    
    # === Write to Gold Label Store ===
    partitions = df.select("label_store_partition").distinct().rdd.map(lambda r: r[0]).collect()
    for partition in partitions:
        subset = df.filter(df["label_store_partition"] == partition)
        output_path = f"{gold_dir}/label_store/weeks/{partition}"
        logging.info(f"Writing gold label store for {partition} to {output_path}")
        subset.write.mode("overwrite").parquet(output_path)

    logging.info("Gold label store created successfully.")
    print(f"Gold label store written to: {gold_dir}")