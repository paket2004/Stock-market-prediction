import os
import subprocess
import pandas as pd
import hydra
import zipfile
import yaml

from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig


def get_increment_counter(path):
    with open(path, 'r') as counter_file:
        counter_data = yaml.safe_load(counter_file)
        counter = counter_data.get('file_version', 0)
    
    new_counter = counter + 1 if counter < 4 else 0

    with open(path, 'w') as counter_file:
        yaml.safe_dump({'file_version': new_counter}, counter_file)

    return counter

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra.main(version_base=None, config_path=f"{project_root_dir}/configs", config_name="config")
def sample_data(cfg: DictConfig = None):

    counter_file_path = os.path.join(project_root_dir, cfg.batch.counter_file)
    kaggle_executable = f'{project_root_dir}/.venv/bin/kaggle'
    command = [kaggle_executable, 'datasets',  'download',  '-d', cfg.dataset.url, '-p', os.path.join(project_root_dir, 'temp')]
    print(command)

    subprocess.run(command)
    zip_file_path = f'{project_root_dir}/temp/{cfg.dataset.archive_name}'
    extract_to_path = f'{project_root_dir}/temp/data'

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)

    file_name = cfg.dataset.file_name
    df = pd.read_csv(os.path.join(extract_to_path, file_name))

    df = df.sample(frac=1, random_state=cfg.batch.random_seed).reset_index(drop=True)
    counter = get_increment_counter(counter_file_path)
    batch_size = cfg.batch.size
    batch = df[counter*batch_size:(counter+1)*batch_size]

    batch.to_csv(os.path.join(project_root_dir, cfg.batch.save_dir, f'sample.csv'))
    
    print(counter)
    return counter

