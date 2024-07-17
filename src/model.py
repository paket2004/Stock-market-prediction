import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import mlflow.exceptions

def train(train_data: pd.DataFrame, model_config: dict):
    X_train = train_data.drop(columns='close')
    y_train = train_data['close']

    if model_config['type'] == 'gradient_boosting':
        model = GradientBoostingRegressor(
            learning_rate=model_config['hyperparameters']['learning_rate'],
            n_estimators=model_config['hyperparameters']['n_estimators'],
            max_depth=model_config['hyperparameters']['max_depth']
        )
    elif model_config['type'] == 'mlp':
        model = MLPRegressor(
            hidden_layer_sizes=model_config['hyperparameters']['hidden_layer_sizes'],
            activation=model_config['hyperparameters']['activation'],
            solver=model_config['hyperparameters']['solver'],
            learning_rate_init=model_config['hyperparameters']['learning_rate_init'],
            max_iter=model_config['hyperparameters']['max_iter'],
        )
    else:
        raise ValueError("Unsupported model type: {}".format(model_config['type']))

    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(model, "model")

    return model


def evaluate(model, val_data: pd.DataFrame):
    X_val = val_data.drop(columns='close')  # Replace 'close' with your actual target column name
    y_val = val_data['close']  # Replace 'close' with your actual target column name

    y_pred = model.predict(X_val)

    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    accuracy = np.mean(np.abs((y_val - y_pred) / y_val) < 0.1)  # 10% margin

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "accuracy": accuracy
    }



def log_metadata(metrics, model, params, prefix):
    experiment_name = "model"
    try:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(run_name="run-01", experiment_id=experiment_id) as run:

        mlflow.log_params(params=params)

        mlflow.log_metrics({
            "accuracy": metrics['accuracy'],
            "mse": metrics['mse'],
            "mae": metrics['mae'],
            "r2": metrics['r2']
        })

        mlflow.set_tag(f"{prefix} Info", f"{model['type']} model for my data")


        mlflow.sklearn.log_model(model, 'regression_model')
