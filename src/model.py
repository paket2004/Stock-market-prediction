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
import giskard
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



def train(train_data: pd.DataFrame, validate_data: pd.DataFrame, model_config: dict):

    best_models = dict()

    X_train = train_data.drop(columns='Adj Close')
    y_train = train_data['Adj Close']
    
    param_grid = list(product(*model_config['hyperparameters'].values()))
    param_names = list(model_config['hyperparameters'].keys())
    
    best_score = float('inf')
    best_model = None
    best_params = None
    
    kf = KFold(n_splits=3)


    mlflow.sklearn.autolog(disable=True)  

    for params in tqdm(param_grid):
        param_dict = dict(zip(param_names, params))
        if model_config['type'] == 'gradient boosting':
            model = GradientBoostingRegressor(random_state=random_seed, **param_dict)
            print('starting gradient boosting ...')
        
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
        
        avg_score = np.mean(fold_scores)
        best_models[model] = [avg_score, param_dict]


        evaluate(model, model_config['type'], validate_data, 'train', param_dict)

    best_models = dict(sorted(best_models.items(), key=lambda item: item[1][0]))

    return best_models
            


def evaluate(model, model_type, val_data, context, params=None):
    X_val = val_data.drop(columns='Adj Close') 
    y_val = val_data['Adj Close'] 

    y_pred = model.predict(X_val)

    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    accuracy = np.mean(np.abs((y_val - y_pred) / y_val) < 0.05)  # 5% margin    

    metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "accuracy": accuracy
    }
    log_metadata(metrics, model, model_type, X_val, y_val, y_pred, context, params)
    # return {
    #     "mse": mse,
    #     "mae": mae,
    #     "r2": r2,
    #     "accuracy": accuracy
    # }, X_val, y_val, y_pred



def log_metadata(metrics, model, model_type, X, y_val, y_pred, context, params=None):
    print('\n\nparams:\n', params, '\n\n')

    eval_data = X.copy()
    eval_data["true"] = y_val

    # Assign the decoded predictions to the Evaluation Dataset
    eval_data["predictions"] = y_pred

    # Create the PandasDataset for use in mlflow evaluate
    pd_dataset = mlflow.data.from_pandas(
        eval_data, predictions="predictions", targets="true"
    )

    mlflow.set_tracking_uri(uri="http://localhost:5000")
    experiment_name = "Stock Market Prediction"
    try:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.sklearn.autolog(disable=True)

    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run(run_name=f"{model_type} {params.values() if params else ''}, {context}", experiment_id=experiment_id) as run:

        mlflow.log_input(pd_dataset, context=context)
        
        if params:
            mlflow.log_params(params=params)

        mlflow.log_metrics({
            "accuracy": metrics['accuracy'],
            "mse": metrics['mse'],
            "mae": metrics['mae'],
            "r2": metrics['r2']
        })

        mlflow.set_tag(f"{context} info", f"{model_type} model for my data")
        print()
        print(X.columns)
        mlflow.sklearn.log_model(model, 'regression_model', input_example=X)

        result = mlflow.evaluate(data=pd_dataset, 
                                 model_type='regressor', 
                                 evaluators = ["default"])


        plot_and_log_metrics(y_val, y_pred, 0)




def retrieve_model_with_alias(model_name, model_alias):

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
