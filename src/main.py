import hydra
import pandas as pd
from omegaconf import DictConfig
from data import extract_data, split_data
from model import train, evaluate, log_metadata
import yaml
import mlflow
from mlflow.tracking import MlflowClient
import os

random_seed = 42


def get_version():
    file_path = '../configs/data_version.yaml'
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['file_version']


def download_artifacts(dst_path='../results'):
    client = MlflowClient()

    experiment_name = "Stock Market Prediction"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    
    # Get all runs from the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    
    # Download artifacts for each run
    for run_id, run_name in zip(runs['run_id'], runs['run_name']):

        print(run_id)
        print(client.list_artifacts(run_id))

        plot_folder = os.path.join(dst_path, run_name)
        os.makedirs(plot_folder)

        client.download_artifacts(run_id, 'performance.png', dst_path=plot_folder)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print('started main.py...')

    train_version = 0 #get_version()
    test_version = train_version+1 if train_version < 4 else 0

    data = extract_data(str(train_version))


    train_data, val_data = split_data(data, cfg.split_ratio)

    test_data = extract_data(str(test_version))

    model, params, _ = train(train_data, cfg.model)

    val_metrics, y_val, y_pred = evaluate(model, val_data)

    log_metadata(val_metrics, model, params, cfg.model.type, y_val, y_pred, prefix='Val')

    test_metrics, y_val, y_pred = evaluate(model, test_data)

    log_metadata(test_metrics, model, params, cfg.model.type, y_val, y_pred, prefix='Test')

    download_artifacts()

if __name__ == "__main__":
    main()
