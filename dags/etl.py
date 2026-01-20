"""
Airflow DAG for ML Pipeline: Data Ingestion → Transformation → Model Training (PyTorch) + DVC Integration
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
import os
import sys

# Add project root to PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Default arguments
default_args = {
    'owner': 'ml_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
}

# DAG definition
dag = DAG(
    'ml_pipeline_dvc_pytorch',
    default_args=default_args,
    description='DVC-based ML Pipeline: Ingestion → Transformation → Model Training (PyTorch + HPO)',
    schedule='@weekly',
    catchup=False,
    tags=['ml', 'pytorch', 'dvc', 'pipeline']
)

# Base path for logs and project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

extract_task = BashOperator(
    task_id="extract",
    bash_command=f"cd {PROJECT_ROOT} && dvc repro -s data_ingestion >> {LOGS_DIR}/extract.log 2>&1",
    dag=dag,
)

transform_task = BashOperator(
    task_id="transform",
    bash_command=f"cd {PROJECT_ROOT} && dvc repro -s data_transformation >> {LOGS_DIR}/transform.log 2>&1",
    dag=dag,
)

load_task = BashOperator(
    task_id="load",
    bash_command=f"cd {PROJECT_ROOT} && dvc repro -s model_training >> {LOGS_DIR}/load.log 2>&1",
    dag=dag,
)

push_artifacts_task = BashOperator(
    task_id="push_artifacts",
    bash_command=f"cd {PROJECT_ROOT} && dvc push >> {LOGS_DIR}/dvc_push.log 2>&1",
    dag=dag,
)

# Set ETL execution order
extract_task >> transform_task >> load_task >> push_artifacts_task