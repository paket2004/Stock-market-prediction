from zenml.pipelines import pipeline
from typing import Annotated, Optional, Tuple

from zenml.steps import step
import os
import yaml
import pandas as pd

import data as dt

project_root_dir = '/home/user/Stock-market-prediction'

@step(enable_cache=False)
def extract_data_step(
                     ) -> Tuple[Annotated[pd.DataFrame, 'data'], 
                                Annotated[str, 'version']]:

    print(os.path.abspath(__file__))
    df = dt.read_datastore()
    with open(f"{project_root_dir}/configs/data_version.yaml", 'r') as file:
        config = yaml.safe_load(file)
        data_version = config['file_version']

    return df, str(data_version)

@step(enable_cache=False)
def transform_data_step(
                        data: pd.DataFrame
                        ) -> Tuple[Annotated[pd.DataFrame, 'X'], 
                                   Annotated[pd.DataFrame, 'y']]:

    X, y = dt.preprocess_data(data)
    return X, y

@step(enable_cache=False)
def validate_data_step(
                        X: pd.DataFrame, y: pd.DataFrame
                        ) -> Tuple[Annotated[pd.DataFrame, 'X'],
                                   Annotated[pd.DataFrame, 'y']]:

    X, y = dt.validate_features(X, y)
    return X, y

@step(enable_cache=False)
def load_data_step(X: pd.DataFrame, y: pd.DataFrame, version: str):

    dt.load_features(X, y, version)



@pipeline
def data_prepare_pipeline(
    extract_data_step,
    transform_data_step,
    validate_data_step,
    load_data_step,
):
    data, version = extract_data_step()
    X, y = transform_data_step(data=data)
    X, y = validate_data_step(X=X, y=y)
    load_data_step(X=X, y=y, version=version)


if __name__ == "__main__":
    extract_data = extract_data_step()
    transform_data = transform_data_step()
    validate_data = validate_data_step()
    load_data = load_data_step()

    p = data_prepare_pipeline(
        extract_data_step=extract_data,
        transform_data_step=transform_data,
        validate_data_step=validate_data,
        load_data_step=load_data,
    )

    p.run()