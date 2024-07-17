from pendulum import datetime
from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from airflow.utils.dates import days_ago
from datetime import timedelta
import os

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dag(    
    dag_id ='data_prepare_dag',
    description='A DAG to run data preparation pipeline after data extraction',
    start_date=datetime(2024, 6, 30, tz='UTC'),
    schedule='*/5 * * * *',
    catchup=False
)

def data_prepare():

    wait_for_data_extraction = ExternalTaskSensor(
        task_id='wait_for_data_extraction',
        external_dag_id='data_extract',
        external_task_id=None,  # wait for the entire DAG to complete
        check_existence=True,
        mode='poke',
        poke_interval=60,
        timeout=600,
    )

    run_data_prepare_pipeline = BashOperator(
        task_id='run_data_prepare_pipeline',
        bash_command=f'python3 /home/user/Stock-market-prediction/pipelines/data_prepare.py -data_prepare_pipeline',
    )

    wait_for_data_extraction >> run_data_prepare_pipeline


data_prepare()