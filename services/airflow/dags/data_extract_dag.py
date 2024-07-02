from pendulum import datetime
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models.baseoperator import chain
import os

project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sample_path = os.path.join(project_root_dir, 'data', 'samples', 'sample.csv')
version_path = os.path.join(project_root_dir, 'connfigs', 'data_version.yaml')

@dag(
    dag_id="data_extract", 
    start_date=datetime(2024, 6, 30, tz='UTC'),
    schedule='*/5 * * * *',
    catchup=False
)
def data_extract():

    extract = BashOperator(task_id='extraction', 
                           bash_command=f'python3 {project_root_dir}/src/data.py',
                           do_xcom_push=True)
    
    validate = BashOperator(task_id='validation',
                            bash_command=f'python3 {project_root_dir}/scripts/validate_initial_data.py')


    def version_data(**context):
        version = context['ti'].xcom_pull(task_ids='extraction')

        os.system(f'dvc add {sample_path}')
        os.system(f'dvc commit -m "Versioning data sample version {version}"')

        os.system(f'git add {sample_path}.dvc {version_path}')
        os.system(f'git commit -m "Versioning data sample version {version}"')
        os.system(f'git tag v{version}')
        os.system(f'git push origin --tags')
        os.system(f'git push')

    version = PythonOperator(task_id='versioning', 
                             python_callable=version_data,
                             provide_context=True)


    load = BashOperator(task_id='loading',
                        bash_command=f'cd {project_root_dir} && dvc push')

    extract >> validate >> version >> load
    

data_extract()