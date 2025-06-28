from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import EmptyOperator
from airflow.operators.python import ShortCircuitOperator
from datetime import datetime, timedelta
from pathlib import Path
import sys


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='assn2',
    default_args=default_args,
    description='data processing and model training pipeline',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True
) as dag:

    # data pipeline

    # check if raw data exists
    check_raw_data = BashOperator(
        task_id='check_raw_data',
        bash_command=(
            'test -f /opt/airflow/scripts/data/feature_clickstream.csv && '
            'test -f /opt/airflow/scripts/data/features_attributes.csv && '
            'test -f /opt/airflow/scripts/data/features_financials.csv && '
            'test -f /opt/airflow/scripts/data/lms_loan_daily.csv'
        )
    )

    # --- Bronze Processing ---
    bronze_process = BashOperator(
        task_id='bronze_processing',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_processing.py '
            '--snapshotdate "{{ ds }}"'
            ),
    )

    # --- Silver Processing ---
    silver_process = BashOperator(
        task_id='silver_processing',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_processing.py '
            '--snapshotdate "{{ ds }}"'
            ),
    )

    # --- Gold Processing ---

    gold_feature_store = BashOperator(
        task_id="process_gold_features",
        bash_command="python3 /opt/airflow/scripts/gold_processing.py --snapshotdate {{ ds }} --task features",
    )

    gold_label_store = BashOperator(
        task_id="process_gold_labels",
        bash_command="python3 /opt/airflow/scripts/gold_processing.py --snapshotdate {{ ds }} --task labels",
    )

    # --- Data Processing Complete ---
    feature_store_complete = EmptyOperator(task_id='feature_store_complete')
    label_store_complete = EmptyOperator(task_id='label_store_complete')

    # Defining task dependencies
    check_raw_data >> bronze_process 
    bronze_process >> silver_process 
    silver_process >> [gold_feature_store, gold_label_store] 
    gold_feature_store >> feature_store_complete
    gold_label_store >> label_store_complete

    # model pipeline

    # --- Model Training ---
    
    # Run gold processing for training window
    gold_for_training = BashOperator(
        task_id='gold_for_training',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 run_gold_for_training.py '
            '--startdate {{ ["2023-01-01", macros.ds_add(ds, -365)] | max }} '
            '--enddate {{ ds }}'
        ),
    )
    
    # Check if we have at least 1 year of data for training
    def check_sufficient_data(**context):
        execution_date = context['execution_date']
        earliest_data_date = datetime(2023, 1, 1).date()
        training_start_date = max(earliest_data_date, execution_date.date() - timedelta(days=365))
        
        # We have sufficient data if the training window is at least 1 year
        # This means training_start_date should be at least 365 days before execution_date
        return (execution_date.date() - training_start_date).days >= 365
    
    check_data_sufficiency = ShortCircuitOperator(
        task_id='check_data_sufficiency',
        python_callable=check_sufficient_data,
    )
    
    train_model = BashOperator(
        task_id='train_model',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_training.py '
            '--startdate {{ ["2023-01-01", macros.ds_add(ds, -365)] | max }} '
            '--enddate {{ ds }} '
            '--modelname model_{{ ds_nodash }}'
        ),
    )
    
    # --- Model Inference ---
    model_inference = BashOperator(
        task_id="model_inference",
        bash_command="/opt/airflow/scripts/model_inf.sh {{ ds }} {{ ds_nodash }}"
    )

    # --- Monitoring ---
    model_monitoring = EmptyOperator(
        task_id="model_monitoring" #,
        # bash_command="/opt/airflow/scripts/model_monitor.sh {{ ds }} {{ ds_nodash }}"
    )
    model_inference >> model_monitoring

    # task dependencies
    [feature_store_complete, label_store_complete] >> check_data_sufficiency
    check_data_sufficiency >> gold_for_training
    gold_for_training >> train_model
    train_model >> model_inference >> model_monitoring