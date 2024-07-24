import os
import sys
import pytest
import pandas as pd
import yaml
import numpy as np
from unittest import mock
from unittest.mock import patch, MagicMock

dir_name = os.path.dirname(os.path.abspath(__file__))

project_root_dir = os.path.dirname(dir_name)

# __init__.py in the src folder must be
sys.path.insert(0, os.path.abspath(os.path.join(project_root_dir, 'src')))

from evaluate import evaluate

@patch('evaluate.mlflow.set_tracking_uri')
@patch('evaluate.retrieve_model_with_alias')
@patch('evaluate.extract_data')
@patch('builtins.print')
def test_evaluate(mock_print, mock_extract_data, mock_retrieve_model_with_alias, mock_set_tracking_uri):
    # mocking
    mock_set_tracking_uri.return_value = None
    mock_model = MagicMock()
    mock_model.predict.return_value = [3.0, 2.5, 4.0, 5.0]

    # mock the retrieve_model_with_alias function
    mock_retrieve_model_with_alias.return_value = (mock_model, '1.0')

    # custom dataframe and mocked extract_data function 
    mock_data = pd.DataFrame({
        'Feature1': [1.0, 2.0, 3.0, 4.0],
        'Feature2': [2.0, 3.0, 4.0, 5.0],
        'Adj Close': [3.0, 2.0, 4.0, 5.0]
    })
    mock_extract_data.return_value = mock_data

    evaluate(model_alias='champion', model_name='boosting_01_300_4', data_sample_version=0)

    # check if the print statements were called with the correct evaluation metrics
    mock_print.assert_any_call('Model Alias: champion')
    mock_print.assert_any_call('Mean Squared Error: 0.0625')
    mock_print.assert_any_call('Mean Absolute Error: 0.125')
    mock_print.assert_any_call('R2 Score: 0.95')
