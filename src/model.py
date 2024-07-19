import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import mlflow.exceptions
from itertools import product
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
random_seed = 42
np.random.seed(random_seed)


def plot_and_log_metrics(y_true, y_pred, fold):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r', linewidth=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted - Fold {fold}')
    plt.grid(True)
    
    image_path = f'performance.png'
    plt.savefig(image_path)
    plt.close()
    
    mlflow.log_artifact(image_path)
    os.remove(image_path)



def train(train_data: pd.DataFrame, model_config: dict):
    X_train = train_data.drop(columns='Adj Close')
    y_train = train_data['Adj Close']
    
    param_grid = list(product(*model_config['hyperparameters'].values()))
    param_names = list(model_config['hyperparameters'].keys())
    
    best_score = float('inf')
    best_model = None
    best_params = None
    
    kf = KFold(n_splits=3)

    mlflow.set_tracking_uri(uri="http://localhost:5000")
    experiment_name = "Stock Market Prediction"
    try:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.sklearn.autolog(disable=True)  

    for params in tqdm(param_grid):
        param_dict = dict(zip(param_names, params))
        if model_config['type'] == 'gradient boosting':
            model = GradientBoostingRegressor(random_state=random_seed, **param_dict)
        elif model_config['type'] == 'lightgbm':
            model = LGBMRegressor(random_state=random_seed, **param_dict)
            print('starting lightGBM ...')
        else:
            raise ValueError("Unsupported model type: {}".format(model_config['type']))
        
        fold_scores = []
        for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
            
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            fold_score = mean_squared_error(y_fold_val, y_pred)
            fold_scores.append(fold_score)
            
            # Log the model for this fold
        with mlflow.start_run(run_name=f"{model_config['type']} {params}", experiment_id=experiment_id) as run:
            mlflow.log_params(param_dict)
            mlflow.log_metric("mse", fold_score)
            mlflow.sklearn.log_model(model, f"model_fold_{fold}")

            plot_and_log_metrics(y_fold_val, y_pred, fold)

        
        avg_score = np.mean(fold_scores)
        if avg_score < best_score:
            best_score = avg_score
            best_model = model
            best_params = param_dict
    
    return best_model, best_params, best_score


def evaluate(model, val_data: pd.DataFrame):
    X_val = val_data.drop(columns='Adj Close') 
    y_val = val_data['Adj Close'] 

    y_pred = model.predict(X_val)

    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    accuracy = np.mean(np.abs((y_val - y_pred) / y_val) < 0.01)  # 10% margin    

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "accuracy": accuracy
    }, y_val, y_pred



def log_metadata(metrics, model, params, model_type, y_val, y_pred, prefix):
    print('\n\nparams:\n', params, '\n\n')

    mlflow.set_tracking_uri(uri="http://localhost:5000")
    experiment_name = "Stock Market Prediction"
    try:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.sklearn.autolog(disable=True)

    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run(run_name=f"best {model_type} model, {prefix}", experiment_id=experiment_id) as run:

        mlflow.log_params(params=params)

        mlflow.log_metrics({
            "accuracy": metrics['accuracy'],
            "mse": metrics['mse'],
            "mae": metrics['mae'],
            "r2": metrics['r2']
        })

        mlflow.set_tag(f"{prefix} Info", f"{model_type} model for my data")

        mlflow.sklearn.log_model(model, 'regression_model')

        plot_and_log_metrics(y_val, y_pred, 0)




def retrieve_model_with_alias(model_name, model_alias):
    """
    Retrieve an MLflow model based on its name and alias.

    Parameters:
    - model_name (str): Name of the MLflow model.
    - model_alias (str): Alias (version alias) of the model.

    Returns:
    - mlflow.pyfunc.PyFuncModel: Loaded PyFuncModel object representing the MLflow model.
    """
    client = MlflowClient()

    # Get model version information by alias
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias=model_alias)
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve model version '{model_alias}' for model '{model_name}': {e}")

    # Get the model URI
    model_uri = mv.source

    # Load the MLflow model
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}' from URI '{model_uri}': {e}")

    return model, mv.version
