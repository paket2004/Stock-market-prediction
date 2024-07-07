import os
import subprocess
import pandas as pd
import zipfile
import yaml

from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, Flatten, Dense
from sklearn.decomposition import PCA
import gensim.downloader as api

import dvc.api


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

        batch.to_csv(os.path.join(project_root_dir, cfg.batch.save_dir, f'sample.csv'))
        
        print(counter)
        return counter


def read_datastore():
    data_path = dvc.api.get_url(f'{project_root_dir}/data/sample_data.csv', 
                                remote='local_remote')
    df = pd.read_csv(data_path)
    return df



def preprocess_data():

    def encode_time (data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
        return data


    def get_average_embedding(text):
        wv = api.load('word2vec-google-news-300')
        words = text.split()
        embedding_vector = np.zeros((300,))
        for word in words:
            if word in wv:
                embedding_vector += wv[word]
        return embedding_vector / len(words)



    def expand_embedding_list(row):
        return pd.Series(row['Symbol emb'])


    #################


    stock_data = pd.read_csv("data/samples/sample.csv")

    numeric_cols = stock_data.select_dtypes(include=["number"]).columns
    correlation_matrix = stock_data[numeric_cols].corr()
    correlation_with_target = correlation_matrix['Close'].abs().sort_values(ascending=False)
    low_correlation = correlation_with_target[correlation_with_target < 0.05]

    stock_data_cleaned = stock_data.drop(['Security', 'Adj Close'], axis=1)
    stock_data_cleaned = stock_data_cleaned.drop(low_correlation.index, axis=1)

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
    stock_data_cleaned = pd.get_dummies(stock_data_cleaned, columns=['GICS Sector'], prefix='Sector', dtype=int)


    symbol_indices = {symbol: idx for idx, symbol in enumerate(stock_data_cleaned['Symbol'].unique())}
    stock_data_cleaned['Symbol_Index'] = stock_data_cleaned['Symbol'].map(symbol_indices)

    embedding_size = 10  # Dimensionality of the embedding vector

    model = Sequential([
        Embedding(len(symbol_indices), embedding_size, input_length=1),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    X = np.array(stock_data_cleaned['Symbol_Index'])
    y = np.array(stock_data_cleaned['Close'])

    model.fit(X, y, epochs=10, batch_size=1)

    learned_embeddings = model.layers[0].get_weights()[0]

    idx_AAPL = symbol_indices['AAPL']
    embedding_AAPL = learned_embeddings[idx_AAPL]
    print(f"Embedding for AAPL: {embedding_AAPL}")

    stock_data_cleaned['Symbol emb'] = np.array(
        [learned_embeddings[idx] for idx in stock_data_cleaned['Symbol_Index']]).tolist()



    expanded_embeddings = stock_data_cleaned.apply(expand_embedding_list, axis=1)
    expanded_embeddings.columns = [f'Symbol emb_{i}' for i in range(len(expanded_embeddings.columns))]

    stock_data_cleaned = pd.concat(
        [stock_data_cleaned.drop(columns=['Symbol emb', 'Symbol_Index', 'Symbol']), expanded_embeddings], axis=1)


    txt_embeddings = stock_data_cleaned['GICS Sub-Industry'].apply(lambda x: get_average_embedding(x))
    txt_embeddings = np.array(txt_embeddings.tolist())

    pca = PCA(n_components=16)
    pca_result = pca.fit_transform(txt_embeddings)

    embedding_cols = pd.DataFrame(pca_result, columns=[f'Emb_{i}' for i in range(16)])

    stock_data_cleaned = pd.concat([stock_data_cleaned.drop('GICS Sub-Industry', axis=1), embedding_cols], axis=1)

    X = stock_data_cleaned.drop("Close", axis=1)
    y = stock_data_cleaned[['Close']]
    return X, y
