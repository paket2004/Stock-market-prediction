from pendulum import datetime
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models.baseoperator import chain

@dag(
    dag_id="data_extract", 
    start_date=datetime(2024, 30, 6, tz='Russia/Moscow'),
    schedule='*/5 * * * *',
    catchup=False
)
def 