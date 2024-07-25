from unittest import mock
import os
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import sys
import os
import pytest
import pandas as pd

dir_name = os.path.dirname(os.path.abspath(__file__))

project_root_dir = os.path.dirname(dir_name)

# __init__.py in the src folder must be
sys.path.insert(0, os.path.abspath(os.path.join(project_root_dir, 'src')))


from model import (
    plot_and_log_metrics,
    train,
    evaluate,
    log_metadata,
    retrieve_model_with_alias
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


@patch('model.plt.savefig')
@patch('model.mlflow.log_artifact')
@patch('model.os.remove')
def test_plot_and_log_metrics(mock_log_artifact, mock_savefig, mock_remove):
    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred = pd.Series([1.1, 1.9, 3.0, 3.9, 5.2])
    fold = 1

    plot_and_log_metrics(y_true, y_pred, fold)

    # Check if savefig and log_artifact were called
    mock_savefig.assert_called_once_with('performance.png')
    mock_log_artifact.assert_called_once_with('performance.png')

    # Check if file removal was done
    mock_remove.assert_called_once_with('performance.png')


@patch('model.KFold')
@patch('model.GradientBoostingRegressor')
@patch('model.LGBMRegressor')
@patch('model.evaluate')
def test_train(mock_evaluate, mock_lgbm, mock_gb, mock_kfold):

    mock_kf = MagicMock()
    mock_kfold.return_value = mock_kf
    
    # mocking the models (Gradient Boosting and LGBM) fit and predict methods
    mock_gb.return_value.fit = MagicMock()
    mock_gb.return_value.predict = MagicMock(return_value=[1, 2, 3])
    mock_lgbm.return_value.fit = MagicMock()
    mock_lgbm.return_value.predict = MagicMock(return_value=[1, 2, 3])
    
    # prepare data
    train_data = pd.DataFrame({'Adj Close': [2, 4, 6, 8, 10], 'Feature': [1, 2, 3, 4, 5]})
    validate_data = pd.DataFrame({'Adj Close': [2, 4, 6], 'Feature': [1, 2, 3]})

    # take a gradient boosting model for this test
    model_config = {
        'type': 'gradient boosting',
        'hyperparameters': {'n_estimators': [10, 50, 70], 'learning_rate': [0.1, 0.01]}
    }
    
    # call the function
    best_models = train(train_data, validate_data, model_config)
    
    # check if evaluate was called (has to be). Also, the dictionary(!) cannot be empty
    mock_evaluate.assert_called()
    assert isinstance(best_models, dict)
    assert len(best_models) > 0



@patch('model.log_metadata')
def test_evaluate(mock_log_metadata):
    # Prepare test data
    val_data = pd.DataFrame({'Adj Close': [2, 4, 6, 8, 10], 'Feature': [1, 2, 3, 4, 5]})
    model = MagicMock()
    model.predict = MagicMock(return_value=[1, 2, 3, 4, 5])
    
    evaluate(model, 'gradient boosting', val_data, 'test_context')
    
    # check if log_metadata was called (in the end) - the result will affect log_metadata
    mock_log_metadata.assert_called()


@patch('model.mlflow.set_tracking_uri')
@patch('model.mlflow.create_experiment')
@patch('model.mlflow.get_experiment_by_name')
@patch('model.mlflow.sklearn.log_model')
@patch('model.mlflow.log_params')
@patch('model.mlflow.log_metrics')
@patch('model.mlflow.log_input')
@patch('model.plot_and_log_metrics')
@patch('model.mlflow.start_run')
def test_log_metadata(mock_start_run, mock_plot_and_log_metrics, mock_log_input, mock_log_metrics, 
                      mock_log_params, mock_log_model,
                      mock_get_experiment_by_name, mock_create_experiment, mock_set_tracking_uri):

    X = pd.DataFrame({'Feature': [1, 2, 3]})
    y_val = [1, 2, 3]
    y_pred = [1, 2, 3]
    model = MagicMock()
    metrics = {"mse": 0.1, "mae": 0.1, "r2": 0.9, "accuracy": 0.95}

    # mocking to simulate the MLflow environment
    mock_set_tracking_uri.return_value = None
    mock_create_experiment.return_value = 1
    mock_get_experiment_by_name.return_value = MagicMock(experiment_id=1)

    mock_start_run.return_value.__enter__ = mock.Mock(return_value=None)
    mock_start_run.return_value.__exit__ = mock.Mock(return_value=None)

    log_metadata(metrics, model, 'gradient boosting', X, y_val, y_pred, 'text_context')
    
    # check ALL MLflow calls (except mlflow.log_params because 'params' was 
    # not passed and mlflow.get_experiment_by_name (try except block))
    mock_set_tracking_uri.assert_called_once()
    mock_create_experiment.assert_called_once_with(name='Stock Market Prediction')
    # mock_log_input.assert_called_once()
    mock_log_metrics.assert_called_once_with({
        "accuracy": metrics['accuracy'],
        "mse": metrics['mse'],
        "mae": metrics['mae'],
        "r2": metrics['r2']
    })
    mock_log_model.assert_called_once()
    mock_plot_and_log_metrics.assert_called()



@patch('model.MlflowClient')
@patch('model.mlflow.pyfunc.load_model')
def test_retrieve_model_with_alias(mock_load_model, mock_MlflowClient):
   
    # mocking to simulate the MLflow environment
    mock_client = MagicMock()
    mock_MlflowClient.return_value = mock_client
    
    mock_model_version = MagicMock(source='model_uri', version='mock_model_version')
    mock_client.get_model_version_by_alias.return_value = mock_model_version
    mock_load_model.return_value = 'mock_model'
    
    model, version = retrieve_model_with_alias('model_name', 'model_alias')

    # assertions
    assert model == 'mock_model'
    assert version == 'mock_model_version'
    mock_client.get_model_version_by_alias.assert_called_with(name='model_name', alias='model_alias')
    mock_load_model.assert_called_with('model_uri')
