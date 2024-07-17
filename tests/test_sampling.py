import pytest
import sys
import os
import pandas as pd
from unittest import mock
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from hydra import initialize, compose
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import sample_data, get_increment_counter

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def mock_cfg(tmpdir):
    cfg = {
        "batch": {
            "counter_file": "counter.txt",
            "random_seed": 42,
            "size": 100,
            "save_dir": str(tmpdir)
        },
        "dataset": {
            "url": "some-dataset-url",
            "archive_name": "dataset.zip",
            "file_name": "data.csv"
        }
    }
    return OmegaConf.create(cfg)



def test_get_increment_counter(tmpdir):
    import yaml

    counter_file = tmpdir.join("counter.yaml")
    data = {'file_version': 0}
    with open(counter_file, 'w') as yaml_file:
        yaml.safe_dump(data, yaml_file, default_flow_style=False)

    for expected in [0, 1, 2, 3, 4, 0]:
        counter = get_increment_counter(counter_file)
        assert counter == expected



from data import sample_data
@mock.patch("data.os.path.join")
@mock.patch("data.os.path.relpath")
@mock.patch("data.initialize")
@mock.patch("data.compose")
@mock.patch("data.subprocess.run")
@mock.patch("data.zipfile.ZipFile")
@mock.patch("data.pd.read_csv")
@mock.patch("data.get_increment_counter")
@mock.patch('data.pd.DataFrame.to_csv')
def test_sample_data(
    mock_to_csv,
    mock_get_increment_counter,
    mock_read_csv,
    mock_ZipFile,
    mock_run,
    mock_compose,
    mock_initialize,
    mock_relpath,
    mock_join,
):
    # Mock return values and behaviors
    mock_join.side_effect = lambda *args: os.path.normpath("/".join(args))
    mock_relpath.return_value = "relative_path"

    mock_initialize.return_value.__enter__ = mock.Mock(return_value=None)
    mock_initialize.return_value.__exit__ = mock.Mock(return_value=None)   
    from types import SimpleNamespace

    def dict_to_namespace(d):
        return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})

    mock_compose.return_value = dict_to_namespace({ 
        'batch': {
            'counter_file': 'counter.yaml',
            'random_seed': 42,
            'size': 10,
            'save_dir': 'saved_batches',
        },
        'dataset': {
            'url': 'some_dataset_url',
            'archive_name': 'archive.zip',
            'file_name': 'data.csv',
        },
    })

    mock_get_increment_counter.return_value = 1

    sample_df = pd.DataFrame({
        'column1': range(100),
        'column2': range(100, 200),
    })
    mock_read_csv.return_value = sample_df

    mock_run.return_value = None
    mock_zip_instance = mock.Mock()
    mock_ZipFile.return_value.__enter__.return_value = mock_zip_instance

    # Call the function
    counter = sample_data()

    # Assertions
    assert counter == 1
    mock_initialize.assert_called_once_with(config_path="relative_path")
    mock_compose.assert_called_once_with(config_name="config")
    mock_get_increment_counter.assert_called_once()
    mock_read_csv.assert_called_once()
    mock_run.assert_called_once()
    mock_ZipFile.assert_called_once()
    mock_zip_instance.extractall.assert_called_once()

    # Check if the sample.csv was created correctly
    mock_read_csv_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
    batch = mock_read_csv_df[10:20]
    batch.to_csv.assert_called_once_with(os.path.join("/home/user/Stock-market-prediction", 'saved_batches', 'sample.csv'), index=False)





from data import read_datastore  
@mock.patch('data.dvc.api.get_url')
@mock.patch('data.pd.read_csv')
def test_read_datastore(mock_read_csv, mock_get_url):

    mock_get_url.return_value = '/mock/path/to/sample.csv'

    sample_df = pd.DataFrame({
        'column1': [1, 2, 3],
        'column2': ['a', 'b', 'c']
    })

    mock_read_csv.return_value = sample_df

    result_df = read_datastore()

    mock_get_url.assert_called_once_with('data/samples/sample.csv', repo='/home/user/Stock-market-prediction', remote='local_remote')
    mock_read_csv.assert_called_once_with('/mock/path/to/sample.csv')
    pd.testing.assert_frame_equal(result_df, sample_df)


