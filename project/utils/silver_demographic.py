import logging
import pandas as pd

def process_demographic(path):
    df = pd.read_parquet(path)
    logging.info(f"Loaded demmographic data with shape {df.shape}")

    # Drop Features Marked In Red According to Data Dictionary File
    drop_columns = [
        "annual_inc_joint", "verification_status_joint", "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", "sec_app_mort_acc", 
        "sec_app_open_acc", "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts"
    ]
    df.drop(columns=drop_columns, inplace=True, errors='ignore')

    # Fill Empty/Missing Values
    df['emp_title'] = (
        df['emp_title']
        .fillna('MISSING')    # Fill NA
        .str.strip()    # Strip acc. to notes in Data Dict file
        .str.upper()    # Uppercase acc. to notes in Data Dict file
    )
    df['emp_length'] = df['emp_length'].fillna('MISSING')
    df['home_ownership'] = df['home_ownership'].fillna('MISSING')

    logging.info("Demographic processing complete")
    return df