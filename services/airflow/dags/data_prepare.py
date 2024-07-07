from zenml.pipelines import pipeline
from zenml.steps import step
import yaml
import pandas as pd
from data import read_datastore

@step
def extrat_data() -> tuple[pd.DataFrame, str]:

    df = read_datastore()
    with open("./configs/data_version.yaml", 'r') as file:
        config = yaml.safe_load(file)
        data_version = config['data_version']

    return df, data_version