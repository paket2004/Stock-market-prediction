# for preprocess_data test, I created a folder Stock-Market-Prediction/temp, since preprocess_data in data.py 
# saves a dataframe in the /temp folder

import os
import sys
import pytest
import pandas as pd
import yaml
import numpy as np
from unittest import mock
from io import StringIO

dir_name = os.path.dirname(os.path.abspath(__file__))

project_root_dir = os.path.dirname(dir_name)

# __init__.py in the src folder must be
sys.path.insert(0, os.path.abspath(os.path.join(project_root_dir, 'src')))

from data import (
    get_increment_counter,
    sample_data,
    read_datastore,
    preprocess_data,
    validate_features,
    load_features,
    extract_data,
    split_data
)

# mocking the os.path.dirname and os.path.abspath
@pytest.fixture
def mock_os_path(monkeypatch):
    def mock_dirname():
        return '/mocked/path'

    def mock_abspath():
        return '/mocked/abs/path'
    
    monkeypatch.setattr(os.path, 'dirname', mock_dirname)
    monkeypatch.setattr(os.path, 'abspath', mock_abspath)

# test get_increment_counter
def test_get_increment_counter():

    # create a new counter.yaml file for testing in the same folder:
    counter_path = os.path.join(dir_name, 'counter.yaml')

    with open(counter_path, 'w') as file:
        yaml.safe_dump({'file_version': 2}, file)
    
    # Run the function
    new_counter = get_increment_counter(counter_path)
    new_counter = get_increment_counter(counter_path)
    
    # Check the result
    with open(counter_path, 'r') as file:
        data = yaml.safe_load(file)
        
    assert new_counter == 3
    assert data['file_version'] == 4

    os.remove(counter_path)


# test the read_datastore
def test_read_datastore():
    """ I don't completely understand this method, I don't have 
    Stock-market-prediction/data/samples/files/md5/37/1802a7d7cc7804045d17a6179b1aba file and cannot invoke this
    function without mock.

    Please, revise this test!
    It is passed, written by GPT, but I'm not sure its correct
    """

    data_path = '/mocked/path/data/samples/sample.csv.dvc'
    mock_df = pd.DataFrame({'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']})

    with mock.patch('dvc.api.get_url', return_value=data_path):
        with mock.patch('pandas.read_csv', return_value=mock_df):
            df = read_datastore()
            assert df.equals(mock_df)


# test sample_data
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
    batch.to_csv.assert_called_once_with(os.path.join(project_root_dir, 'saved_batches', 'sample.csv'), index=False)



@mock.patch("data.os.path.join")
@mock.patch("data.os.path.relpath")
@mock.patch("data.initialize")
@mock.patch("data.compose")
@mock.patch("data.subprocess.run")
@mock.patch("data.zipfile.ZipFile")
@mock.patch("data.pd.read_csv")
@mock.patch("data.get_increment_counter")
@mock.patch("builtins.open", new_callable=mock.mock_open)
@mock.patch('data.pd.DataFrame.to_csv')
def test_sample_data_two(
    mock_to_csv,
    mock_open,
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

    # Mock configurations and paths
    mock_config = mock.MagicMock()
    mock_config.dataset.url = 'dataset_url'
    mock_config.dataset.archive_name = 'archive.zip'
    mock_config.dataset.file_name = 'data.csv'
    mock_config.batch.counter_file = 'counter.yaml'
    mock_config.batch.random_seed = 42
    mock_config.batch.size = 10
    mock_config.batch.save_dir = 'data/samples'
    mock_compose.return_value = mock_config

    mock_get_increment_counter.return_value = 1

    csv_data = "col1,col2\n1,a\n2,b\n3,c\n4,d\n5,e\n6,f\n7,g\n8,h\n9,i\n10,j\n11,k\n12,l"
    df = pd.read_csv(StringIO(csv_data))
    mock_read_csv.return_value = df
    
    mock_run.return_value = None
    mock_zip_instance = mock.Mock()
    mock_ZipFile.return_value.__enter__.return_value = mock_zip_instance

    # Call the function
    counter = sample_data()
    

    # assertions
    mock_initialize.assert_called_once()
    mock_compose.assert_called_once_with(config_name='config')

    # check if it was called actually once
    mock_get_increment_counter.assert_called_once_with(os.path.normpath(os.path.join(project_root_dir, 'counter.yaml')))
    mock_run.assert_called_once_with([
        os.path.join(project_root_dir, '.venv/bin/kaggle'), 'datasets', 'download', '-d', 'dataset_url',
        '-p', os.path.normpath(os.path.join(project_root_dir, 'temp'))
    ])

    # check zip files
    mock_ZipFile.assert_called_once_with(os.path.normpath(os.path.join(project_root_dir,"temp/archive.zip")), 'r')
    mock_zip_instance.extractall.assert_called_once_with(os.path.normpath(os.path.join(project_root_dir,"temp/data")))
    
    mock_join.assert_any_call(project_root_dir, "counter.yaml")
    mock_join.assert_any_call(project_root_dir, "temp")
    mock_join.assert_any_call(os.path.join(project_root_dir, "temp", "data"), "data.csv")
    mock_join.assert_any_call(project_root_dir, "data/samples", "sample.csv")

    assert counter == 1
    assert mock_read_csv.call_count == 2



def test_preprocess_data():

    news_columns = ['News - Dividends', 'News - Corporate Earnings',
                       'News - Personnel Changes', 'News - Mergers & Acquisitions',
                       'News - Product Recalls', 'News - Layoffs', 'News - Stock Rumors',
                       'News - Stocks', 'News - All News Volume', 'News - Analyst Comments']
    data = {col: ['Some news'] * 30 for col in news_columns}
    data['Date'] = pd.date_range("2020-10-01","2020-10-30",freq='d')
    data['Symbol'] = ['MMM'] * 30
    data["GICS Sector"] = list(["Tech", "Finance"] * 15)
    data["GICS Sub-Industry"] = list(["Software", "Banking", 'Industrials'] * 10)
    data["Adj Close"] = [150, 160] * 15
    data["Security"] = ['3M'] * 30
    data["Close"] = [160] * 30

    # custom dataset
    df = pd.DataFrame(data)

    # create /temp:
    temp_path = os.path.join(project_root_dir, 'temp')
    if not os.path.exists(temp_path):
        os.mkdir(os.path.join(project_root_dir, 'temp'))

    # mock gensim model
    mock_wv = mock.MagicMock()
    mock_wv.__contains__.side_effect = lambda key: True
    mock_wv.__getitem__.side_effect = lambda key: np.ones(300)

    with mock.patch('gensim.models.KeyedVectors.load', return_value=mock_wv):
        X, y = preprocess_data(df)
        
        assert "Adj Close" not in X.columns
        for col in news_columns:
            assert col not in X.columns
        features = ['Security', 'Close', 'Date', 'Symbol', 'GICS Sub-Industry']
        for feature in features:
            assert feature not in X.columns
        for i in range(16):
            assert f'Emb_{i}' in X.columns
        assert len(y) == 30
        assert X.isnull().sum().sum() == 0


# test the validate_features
def test_validate_features():
    X = pd.DataFrame({'column1': [1, 2], 'column2': [11, 22]})
    y = pd.DataFrame({'Adj Close': [111, 222]})
    
    mock_context = mock.MagicMock()
    # simulate a successful checkpoint run
    mock_context.run_checkpoint.return_value = {"success": True}

    with mock.patch('great_expectations.data_context.FileDataContext', return_value=mock_context):
        # validate_features uses mock_context instead of the FileDataContext
        X_val, y_val = validate_features(X, y)
        
        assert X_val.equals(X)
        assert y_val.equals(y)


# test the load_features
def test_load_features():
    X = pd.DataFrame({'column1': [1, 2], 'column2': [11, 22]})
    y = pd.DataFrame({'Adj Close': [100, 200]})
    version = "1"
    
    with mock.patch('zenml.save_artifact') as mock_save_artifact:
        # mock_save_artifact instead of the zenml.save_artifact

        load_features(X, y, version)

        # check if the zenml.save_artifact was called once - ensures that the function 
        # is attempting to save the artifact as expected
        mock_save_artifact.assert_called_once()


# test extract_data
def test_extract_data():
    # the logic is the same as in the test_load_features

    mock_df = pd.DataFrame({'column1': [1, 2], 'col2': [11, 22]})
    version = "1"

    with mock.patch('zenml.load_artifact', return_value=mock_df):
        data = extract_data(version)
        assert data.equals(mock_df)


# test split_data
def test_split_data():
    # custom dataframe and ratio of the train is 0.8
    data = pd.DataFrame({'column1': range(1000), 'Adj Close': range(1000)})
    split_ratio = {'train': 0.8, 'val': 0.2}

    train_set, val_set = split_data(data, split_ratio)

    assert len(train_set) == 800
    assert len(val_set) == 200



@mock.patch("data.os.path.join")
@mock.patch("data.os.path.relpath")
@mock.patch("data.initialize")
@mock.patch("data.compose")
@mock.patch("data.subprocess.run")
@mock.patch("data.zipfile.ZipFile")
@mock.patch("data.pd.read_csv")
@mock.patch("data.get_increment_counter")
@mock.patch("builtins.open", new_callable=mock.mock_open)
@mock.patch('data.pd.DataFrame.to_csv')
def test_sample_data_two(
    mock_to_csv,
    mock_open,
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

    # Mock configurations and paths
    mock_config = mock.MagicMock()
    mock_config.dataset.url = 'dataset_url'
    mock_config.dataset.archive_name = 'archive.zip'
    mock_config.dataset.file_name = 'data.csv'
    mock_config.batch.counter_file = 'counter.yaml'
    mock_config.batch.random_seed = 42
    mock_config.batch.size = 10
    mock_config.batch.save_dir = 'data/samples'
    mock_compose.return_value = mock_config

    mock_get_increment_counter.return_value = 1

    csv_data = "col1,col2\n1,a\n2,b\n3,c\n4,d\n5,e\n6,f\n7,g\n8,h\n9,i\n10,j\n11,k\n12,l"
    df = pd.read_csv(StringIO(csv_data))
    mock_read_csv.return_value = df
    
    mock_run.return_value = None
    mock_zip_instance = mock.Mock()
    mock_ZipFile.return_value.__enter__.return_value = mock_zip_instance

    # Call the function
    counter = sample_data()
    

    # assertions
    mock_initialize.assert_called_once()
    mock_compose.assert_called_once_with(config_name='config')

    # check if it was called actually once
    mock_get_increment_counter.assert_called_once_with(os.path.normpath(os.path.join(project_root_dir, 'counter.yaml')))
    mock_run.assert_called_once_with([
        os.path.join(project_root_dir, '.venv/bin/kaggle'), 'datasets', 'download', '-d', 'dataset_url',
        '-p', os.path.normpath(os.path.join(project_root_dir, 'temp'))
    ])

    # check zip files
    mock_ZipFile.assert_called_once_with(os.path.normpath(os.path.join(project_root_dir,"temp/archive.zip")), 'r')
    mock_zip_instance.extractall.assert_called_once_with(os.path.normpath(os.path.join(project_root_dir,"temp/data")))
    
    mock_join.assert_any_call(project_root_dir, "counter.yaml")
    mock_join.assert_any_call(project_root_dir, "temp")
    mock_join.assert_any_call(os.path.join(project_root_dir, "temp", "data"), "data.csv")
    mock_join.assert_any_call(project_root_dir, "data/samples", "sample.csv")

    assert counter == 1
    assert mock_read_csv.call_count == 2
