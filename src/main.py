import hydra
import pandas as pd
from omegaconf import DictConfig
from data import extract_data, split_data
from model import train, evaluate, log_metadata
import yaml


def get_version():
    file_path = '../configs/data_version.yaml'
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['file_version']


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print('started main.py...')

    train_version = 0 #get_version()
    test_version = train_version+1 if train_version < 4 else 0

    data = extract_data(str(train_version))


    train_data, val_data = split_data(data, cfg.split_ratio)

    test_data = extract_data(str(test_version))

    model = train(train_data, cfg.model)

    val_metrics = evaluate(model, val_data)

    log_metadata(val_metrics, model, cfg.model, prefix='Val')

    test_metrics = evaluate(model, test_data)

    log_metadata(test_metrics, model, cfg.model, prefix='Test')
if __name__ == "__main__":
    main()
