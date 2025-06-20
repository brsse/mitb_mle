import logging
import pandas as pd

def process_loan_terms(path):
    df = pd.read_parquet(path)
    logging.info(f"Loaded loan terms data with shape {df.shape}")

    # Drop Features Marked In Red According to Data Dictionary File
    drop_columns = [
        "url", "desc", "title", "hardship_flag", "hardship_type", "hardship_reason", "hardship_status", "deferral_term",
        "hardship_amount", "hardship_start_date", "hardship_end_date", "payment_plan_start_date", "hardship_length",
        "hardship_dpd", "hardship_loan_status", "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount",
        "hardship_last_payment_amount", "debt_settlement_flag_date", "settlement_status", "settlement_date",
        "settlement_amount", "settlement_percentage", "settlement_term", "out_prncp", "out_prncp_inv", "total_pymnt", 
        "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", 
        "last_pymnt_d", "next_pymnt_d", "last_pymnt_amnt", "policy_code"
    ]
    df.drop(columns=drop_columns, inplace=True, errors='ignore')

    # Convert Y/N values to 0/1
    df["pymnt_plan"] = df["pymnt_plan"].map({"y": 1, "n": 0})

    logging.info("Loan terms processing complete")
    return df