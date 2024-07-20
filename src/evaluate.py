import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml
from model import retrieve_model_with_alias
from data import extract_data

def evaluate(model_alias: str, model_name: str, data_sample_version: int):
    mlflow.set_tracking_uri(uri="http://localhost:5000")

    model, version = retrieve_model_with_alias(model_name, model_alias)

    data = extract_data(data_sample_version)
    X = data.drop(columns='Adj Close')
    y_true = data['Adj Close']

    y_pred = model.predict(X)

    # Evaluate metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Model Alias: {model_alias}")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument('--model_alias', type=str, default='champion', help='Model alias to use for evaluation')
    parser.add_argument('--model_name', type=str, default='boosting_01_300_4', help='Name of a model to evaluate')
    parser.add_argument('--data_sample_version', type=int, default=0, help='Data sample version to evaluate')

    args = parser.parse_args()

    evaluate(args.model_alias, args.model_name, args.data_sample_version)
