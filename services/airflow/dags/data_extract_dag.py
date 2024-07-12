from pendulum import datetime
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models.baseoperator import chain
import os
import sys
import subprocess

# sys.path.append(os.path.join(project_root_dir, 'src'))
from data import sample_data



project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sample_path = os.path.join(project_root_dir, 'data', 'samples', 'sample.csv')
version_path = os.path.join(project_root_dir, 'configs', 'data_version.yaml')

@dag(
    dag_id="data_extract", 
    start_date=datetime(2024, 6, 30, tz='UTC'),
    schedule='*/5 * * * *',
    catchup=False
)
def data_extract():

    def call_sample_data():
        sample_data()

    extract = PythonOperator(task_id='extraction', 
                            python_callable=sample_data,
                            do_xcom_push=True)

    
    validate = BashOperator(task_id='validation',
                            bash_command=f'python3 {project_root_dir}/scripts/validate_initial_data.py')


    def version_data(**context):
        version = context['ti'].xcom_pull(task_ids='extraction')

        commands = [
            f'dvc add {sample_path}',
            f'dvc commit',
            f'git add {sample_path}.dvc',
            f'git add {version_path}',
            f'git commit -m "Versioning data sample version {version}"',
            f'git tag v{version}',
            'git push origin --tags',
            'git push'
        ]

        for command in commands:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Command failed: {command}\nOutput: {result.stdout}\nError: {result.stderr}")


    version = PythonOperator(task_id='versioning', 
                             python_callable=version_data,
                             provide_context=True)


    load = BashOperator(task_id='loading',
                        bash_command=f'cd {project_root_dir} && dvc push')

    extract >> validate >> version >> load
    

data_extract()
