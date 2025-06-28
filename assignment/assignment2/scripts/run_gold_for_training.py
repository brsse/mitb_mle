#!/usr/bin/env python3
"""
Script to run gold processing for all dates needed for model training
"""

import argparse
import subprocess
from datetime import datetime, timedelta
import os

def run_gold_processing(start_date, end_date):
    """Run gold processing for all dates in the range"""
    print(f"🔄 Running gold processing for date range: {start_date} to {end_date}")
    
    current_date = start_date
    processed_dates = []
    
    while current_date < end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"📅 Processing: {date_str}")
        
        try:
            # Run gold processing for features
            cmd_features = [
                "python3", "/opt/airflow/scripts/gold_processing.py",
                "--snapshotdate", date_str,
                "--task", "features"
            ]
            result_features = subprocess.run(cmd_features, capture_output=True, text=True)
            
            if result_features.returncode == 0:
                print(f"✅ Features processed for {date_str}")
                
                # Run gold processing for labels
                cmd_labels = [
                    "python3", "/opt/airflow/scripts/gold_processing.py",
                    "--snapshotdate", date_str,
                    "--task", "labels"
                ]
                result_labels = subprocess.run(cmd_labels, capture_output=True, text=True)
                
                if result_labels.returncode == 0:
                    print(f"✅ Labels processed for {date_str}")
                    processed_dates.append(date_str)
                else:
                    print(f"❌ Labels failed for {date_str}: {result_labels.stderr}")
            else:
                print(f"❌ Features failed for {date_str}: {result_features.stderr}")
                
        except Exception as e:
            print(f"❌ Error processing {date_str}: {e}")
        
        # Move to next month
        current_date = current_date + timedelta(days=32)  # Move to next month
        current_date = current_date.replace(day=1)  # Set to first day of month
    
    print(f"\n📊 Summary: Processed {len(processed_dates)} dates")
    if processed_dates:
        print(f"✅ Successfully processed: {', '.join(processed_dates)}")
    
    return processed_dates

def main():
    parser = argparse.ArgumentParser(description="Run gold processing for model training date range")
    parser.add_argument("--startdate", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--enddate", required=True, help="End date YYYY-MM-DD")
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.startdate, "%Y-%m-%d")
    end_date = datetime.strptime(args.enddate, "%Y-%m-%d")
    
    print("🚀 Starting gold processing for model training...")
    processed_dates = run_gold_processing(start_date, end_date)
    
    if processed_dates:
        print(f"\n🎉 Gold processing complete! {len(processed_dates)} dates processed.")
    else:
        print("\n⚠️  No dates were successfully processed.")

if __name__ == "__main__":
    main() 