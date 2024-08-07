import os
import subprocess
import pandas as pd
import zipfile
import yaml

from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import gensim

import zenml
from zenml.client import Client
import dvc.api
import joblib

from sklearn.model_selection import train_test_split

random_seed = 42


def get_increment_counter(path):
    with open(path, 'r') as counter_file:
        counter_data = yaml.safe_load(counter_file)
        counter = counter_data.get('file_version', 0)
    
    new_counter = counter + 1 if counter < 4 else 0

    with open(path, 'w') as counter_file:
        yaml.safe_dump({'file_version': new_counter}, counter_file)

    return counter

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sample_data():
    current_dir = os.path.dirname(__file__)

    file_in_another_directory = os.path.join(current_dir, '..', 'configs')
    relative_path = os.path.relpath(file_in_another_directory, current_dir)

    config_name="config"

    with initialize(config_path=relative_path):
        cfg = compose(config_name=config_name)

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

        batch.to_csv(os.path.join(project_root_dir, cfg.batch.save_dir, f'sample.csv'), index=False)
        
        print('data version: ',counter)
        return counter


def read_datastore(version):

    repo_url = 'https://github.com/paket2004/Stock-market-prediction.git'  # URL to your DVC repository
    rev = f'v{version}' 

    data_path = dvc.api.get_url(f'data/samples/sample.csv', 
                                repo=repo_url,
                                rev=rev,
                                remote='local_remote')
    data_path = project_root_dir+data_path
    df = pd.read_csv(data_path)
    return df




def preprocess_data(df):

    target_variable = 'Adj Close'

    model_path = os.path.join(project_root_dir, 'models', "word2vec-google-news-300.model")
    wv = gensim.models.KeyedVectors.load(model_path)

    def encode_time (data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
        return data


    def get_average_embedding(text):
        words = text.split()
        embedding_vector = np.zeros((300,))
        for word in words:
            if word in wv:
                embedding_vector += wv[word]
        return embedding_vector / len(words)




    #################


    stock_data = df.copy()

    low_correlation = ['News - Dividends', 'News - Corporate Earnings',
                       'News - Personnel Changes', 'News - Mergers & Acquisitions',
                       'News - Product Recalls', 'News - Layoffs', 'News - Stock Rumors',
                       'News - Stocks', 'News - All News Volume', 'News - Analyst Comments']

    stock_data_cleaned = stock_data.drop(['Security', 'Close'], axis=1)
    stock_data_cleaned = stock_data_cleaned.drop(low_correlation, axis=1)

    stock_data_cleaned = stock_data_cleaned.dropna()
    stock_data_cleaned.reset_index(drop=True, inplace=True)


    # Encode cyclic features
    time_info = stock_data_cleaned["Date"]
    time_info = pd.to_datetime(time_info)

    stock_data_cleaned.loc[:, 'Day'] = time_info.dt.day
    stock_data_cleaned = encode_time(stock_data_cleaned, 'Day', 31)

    stock_data_cleaned["Month"] = time_info.dt.month
    stock_data_cleaned = encode_time(stock_data_cleaned, 'Month', 12)

    stock_data_cleaned["Year"] = time_info.dt.year

    stock_data_cleaned = stock_data_cleaned.drop('Month', axis=1)
    stock_data_cleaned = stock_data_cleaned.drop('Day', axis=1)

    stock_data_cleaned = stock_data_cleaned.drop('Date', axis=1)


    # One-hot-encoding on GICS Sector
    all_sector = ['GICS Sector_Communication Services',
       'GICS Sector_Consumer Discretionary', 'GICS Sector_Consumer Staples',
       'GICS Sector_Energy', 'GICS Sector_Financials',
       'GICS Sector_Health Care', 'GICS Sector_Industrials',
       'GICS Sector_Information Technology', 'GICS Sector_Materials',
       'GICS Sector_Real Estate', 'GICS Sector_Utilities']
    

    encoder = joblib.load(f'{project_root_dir}/models/onehot_encoder.pkl')
    categorical = stock_data_cleaned[['GICS Sector']].copy()
    enc_categorical = encoder.fit_transform(categorical)
    stock_data_cleaned = stock_data_cleaned.drop(['GICS Sector'], axis=1)
    encoded_categorical_df = pd.DataFrame(enc_categorical, columns=encoder.get_feature_names_out())
    all_sector_df = pd.DataFrame(0, index=stock_data_cleaned.index, columns=all_sector)

    all_sector_df.update(encoded_categorical_df)
    all_sector_df = all_sector_df.astype('int')

    print(all_sector_df.head())
    stock_data_cleaned = pd.concat([stock_data_cleaned, all_sector_df], axis=1)
    def clean_column_names(df, substring):
        # Create a dictionary with old column names as keys and new column names as values
        new_columns = {col: col.replace(substring, '').strip() 
                       if col !='GICS Sub-Industry' 
                       else col
                       for col in df.columns}
        # Rename the columns
        df.rename(columns=new_columns, inplace=True)

    # Apply the function to remove 'GICS'
    clean_column_names(stock_data_cleaned, 'GICS ')

    # Embedding Symbol

    symbol_indices = {symbol: idx for idx, symbol in enumerate(stock_data_cleaned['Symbol'].unique())}
    stock_data_cleaned['Symbol_Index'] = stock_data_cleaned['Symbol'].map(symbol_indices)

    # Remove Symbol feature
    stock_data_cleaned = stock_data_cleaned.drop(columns=['Symbol'], axis=1)


   #Embedding text feature

    txt_embeddings = stock_data_cleaned['GICS Sub-Industry'].apply(lambda x: get_average_embedding(x))
    txt_embeddings = np.array(txt_embeddings.tolist())

    pca = joblib.load(f'{project_root_dir}/models/pca_model.pkl')

    pca_result = pca.transform(txt_embeddings)

    embedding_cols = pd.DataFrame(pca_result, columns=[f'Emb_{i}' for i in range(16)])

    stock_data_cleaned = pd.concat([stock_data_cleaned.drop('GICS Sub-Industry', axis=1), embedding_cols], axis=1)

    if target_variable in df.columns:
        X = stock_data_cleaned.drop("Adj Close", axis=1)
        y = stock_data_cleaned[['Adj Close']]
    else:
        X = stock_data_cleaned
        y = None

    print('Columns in X:', X.columns.tolist())

    return X, y



def validate_features(X, y):
    from great_expectations.data_context import FileDataContext

    context = FileDataContext(context_root_dir = f"{project_root_dir}/services/gx")

    df = pd.concat([X, y], axis=1)


    batch_request = {
        "runtime_parameters": {"batch_data": df},
        "batch_identifiers": {"default_identifier_name": "default_identifier"},
        "datasource_name": "default_pandas_datasource",
        "data_connector_name": "default_runtime_data_connector_name",
        "data_asset_name": "default_data_asset"
    }



    checkpoint_result = context.run_checkpoint(
        checkpoint_name="feature_val",
        batch_request=batch_request,
    )

    if not checkpoint_result["success"]:
        print("Data validation failed")
        for validation_result in checkpoint_result['run_results'].values():
            result = validation_result['validation_result']
            for expectation_result in result['results']:
                if not expectation_result['success']:
                    print("Failed expectation:")
                    print(expectation_result['expectation_config']['expectation_type'])
                    print("Expectation kwargs:")
                    print(expectation_result['expectation_config']['kwargs'])
                    if "result" in expectation_result and "observed_value" in expectation_result["result"]:
                        observed_value = expectation_result["result"]["observed_value"]
                        print(f"Observed value: {observed_value}")
                    print("-" * 80)
        raise Exception("Data validation failed")
    else:
        print("Data validation passed")

    return X, y

def load_features(X: pd.DataFrame, y: pd.DataFrame, version: str):
    combined_df = pd.concat([X, y], axis=1)

    zenml.save_artifact(data=combined_df, name="feature_target_data", version=version)
    print('saved data v', version)

def extract_data(version: str) -> pd.DataFrame:
    artifact = zenml.load_artifact("feature_target_data", version)

    data = artifact
    for col in data.select_dtypes(include='Int64').columns:
        data[col] = data[col].astype(np.int64)

    return data


def split_data(train_data: pd.DataFrame, split_ratio: dict):

    train_size = split_ratio['train']

    train_set, val_set = train_test_split(train_data, train_size=train_size, random_state=random_seed)

    return train_set, val_set

