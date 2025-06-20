import pandas as pd
import os
import logging
from pathlib import Path
from utils.silver_credit_history import process_credit_history
from utils.silver_demographic import process_demographic
from utils.silver_financial import process_financial
from utils.silver_loan_terms import process_loan_terms

# Set up logging
Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename="logs/silver_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def silver_processing(data_dir, output_dir, data_window=None):
    DATA_DIR = Path(data_dir)
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    categories = {
        "credit_history": process_credit_history,
        "demographic": process_demographic,
        "financial": process_financial,
        "loan_terms": process_loan_terms,
    }

    for category, processor in categories.items():
        logging.info(f"Processing {category} data")
        df = processor(DATA_DIR / f"features_{category}.parquet")

        if data_window is not None:
            data_window_set = set(pd.to_datetime(data_window))
            df = df[df['snapshot_date'].isin(data_window_set)]
        
        category_output_dir = OUTPUT_DIR / category
        category_output_dir.mkdir(parents=True, exist_ok=True)

        for snapshot_date in df['snapshot_date'].unique():
            day_str = pd.to_datetime(snapshot_date).strftime("%Y-%m-%d")
            daily_df = (
                df[df['snapshot_date'] == snapshot_date]
                .copy()
                .sort_values(by=["member_id", "snapshot_date"])
            )
            daily_df = daily_df.groupby("member_id").ffill()

            daily_df.to_parquet(category_output_dir / f"{category}_{day_str}.parquet", index=False)

    logging.info("Silver-level processing complete.")
    print("Silver-level processing complete.")
