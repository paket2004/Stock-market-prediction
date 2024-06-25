import pytest
import sys
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from hydra import initialize, compose
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import sample_data, get_increment_counter

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def mock_cfg(tmpdir):
    cfg = {
        "batch": {
            "counter_file": "counter.txt",
            "random_seed": 42,
            "size": 10,
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
    # Create a temporary file to act as the counter file
    counter_file = tmpdir.join("counter.txt")

    # Initial write to the file
    counter_file.write('0')

    for expected in [0, 1, 2, 3, 4, 0]:
        counter = get_increment_counter(counter_file)
        assert counter == expected


@patch("src.data.subprocess.run")
@patch("src.data.zipfile.ZipFile.extractall")
@patch("src.data.pd.read_csv")
@patch("src.data.get_increment_counter")
def test_sample_data(mock_get_increment_counter, mock_read_csv, mock_extractall, mock_run, mock_cfg, tmpdir):
    mock_get_increment_counter.return_value = 0
    mock_run.return_value = None
    mock_read_csv.return_value = pd.DataFrame({
        "column1": range(100),
        "column2": range(100, 200)
    })

    # Setup temporary directory structure
    temp_dir = tmpdir.mkdir("temp")
    data_dir = temp_dir.mkdir("data")
    save_dir = tmpdir.mkdir("save")

    # Create dataset.zip in /temp directory
    dataset_zip_path = os.path.join(os.path.join(project_root_dir, 'temp', "dataset.zip"))
    with zipfile.ZipFile(dataset_zip_path, 'w') as zipf:
        zipf.write(os.path.join(project_root_dir, 'temp'), arcname=mock_cfg.dataset.archive_name)
    if os.path.exists(dataset_zip_path):
        print('exists')



    mock_zip_file = MagicMock()
    mock_zip_file.__enter__.return_value = mock_zip_file
    mock_extractall.side_effect = lambda extract_to: mock_zip_file.extractall(extract_to)

    # Initialize Hydra with a relative config path and the mock configuration
    with initialize(config_path=None):
        cfg = mock_cfg
        cfg.batch.save_dir = str(save_dir)
        cfg.batch.counter_file = str(tmpdir.join("counter.txt"))

        # Write initial counter file
        counter_file = tmpdir.join("counter.txt")
        counter_file.write("0")

        sample_data(cfg)

        # Check if the Kaggle command was called correctly
        mock_run.assert_called_with([
            f'{project_root_dir}/env/bin/kaggle',
            'datasets', 'download', '-d', cfg.dataset.url, '-p', f'{project_root_dir}/temp'
        ])

        # Check if the extraction was called correctly
        mock_extractall.assert_called_once()

        # Check if the DataFrame was read correctly
        # mock_read_csv.assert_called_once_with(os.path.join(str(data_dir), cfg.dataset.file_name))

        # Check if the output file was created correctly
        output_file = os.path.join(cfg.batch.save_dir, "sample.csv")
        assert os.path.exists(output_file)

        # Check the content of the output file
        output_df = pd.read_csv(output_file)
        assert len(output_df) == cfg.batch.size

    # Clean up: Delete dataset.zip after test completes
    if os.path.exists(dataset_zip_path):
        os.remove(dataset_zip_path)