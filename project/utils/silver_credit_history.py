import logging
import pandas as pd

def process_credit_history(path):
    df = pd.read_parquet(path)
    logging.info(f"Loaded credit history data with shape {df.shape}")

    # Drop Features Marked In Red According to Data Dictionary File
    drop_columns = [
        "last_credit_pull_d", "mths_since_last_record", "mths_since_last_major_derog", "mths_since_recent_bc_dlq", 
        "mths_since_recent_revol_delinq", "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med", 
        "sec_app_mths_since_last_major_derog"
    ]
    df.drop(columns=drop_columns, inplace=True, errors='ignore')
    
    # Add Missing Flags
    missing_flag_columns = ["mort_acc"]

    for col in missing_flag_columns:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # Fill NA for these features with mode
    fill_mode = [
        "earliest_cr_line", "inq_last_6mths", "acc_now_delinq", "delinq_2yrs", "pub_rec", "collections_12_mths_ex_med",
        "chargeoff_within_12_mths", "tax_liens", "pub_rec_bankruptcies", "delinq_amnt"
    ]
    for col in fill_mode:
        mode_val = df[col].mode(dropna=True)[0]
        df[col] = df[col].fillna(mode_val)

    # Fill NA for these features with -1
    fill_neg1 = [
        "inq_last_12m", "inq_fi", "mths_since_last_delinq", "mths_since_recent_inq", "mths_since_rcnt_il", 
        "mths_since_recent_bc", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_accts_ever_120_pd"
    ]
    df[fill_neg1] = df[fill_neg1].fillna(-1)

    logging.info("Credit history processing complete")
    return df