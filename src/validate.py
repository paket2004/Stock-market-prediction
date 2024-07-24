from data import extract_data, preprocess_data # custom module
from model import retrieve_model_with_alias # custom module
import giskard
import hydra
from hydra import initialize, compose
import os
import yaml
import mlflow
from data import read_datastore

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



current_dir = os.path.dirname(__file__)

file_in_another_directory = os.path.join(current_dir, '..', 'configs')
relative_path = os.path.relpath(file_in_another_directory, current_dir)

config_name="config"


with initialize(config_path=relative_path):
    mlflow.set_tracking_uri(uri="http://localhost:5000")

    cfg = compose(config_name=config_name)

    counter_file_path = os.path.join(project_root_dir, cfg.batch.counter_file)


    with open(counter_file_path, 'r') as counter_file:
        counter_data = yaml.safe_load(counter_file)
        version = 0#counter_data.get('file_version', 0)



    df = read_datastore(str(version+1))

    # Specify categorical columns and target column
    TARGET_COLUMN = cfg.data.target_cols[0]

    CATEGORICAL_COLUMNS = list(cfg.data.cat_cols)

    dataset_name = cfg.dataset.name
    

    # Wrap your Pandas DataFrame with giskard.Dataset (validation or test set)
    giskard_dataset = giskard.Dataset(
        df=df,  # A pandas.DataFrame containing raw data (before pre-processing) and including ground truth variable.
        target=TARGET_COLUMN,  # Ground truth variable
        name=dataset_name, # Optional: Give a name to your dataset
        cat_columns=CATEGORICAL_COLUMNS  # List of categorical columns. Optional, but improves quality of results if available.
    )


    best_models = cfg.best_models

    for model_alias, model_name in best_models.items():



        model, model_version = retrieve_model_with_alias(model_name, model_alias = model_alias)  
        print('predicting with model', model_name)

        def predict(raw_df):
            # Transform raw data into features
            X, _ = preprocess_data(df=raw_df)
            # Predict using the model
            predictions = model.predict(X)
            
            return predictions

        predict(df)

        giskard_model = giskard.Model(
            model=predict,
            # A prediction function that encapsulates all the data pre-processing steps and that could be executed with the dataset used by the scan.
            model_type="regression",  # Either regression, classification or text_generation.
            name=model_name,  # Optional.
            feature_names=df.columns,  # Default: all columns of your dataset.
        )

        #scan model
        scan_results = giskard.scan(giskard_model, giskard_dataset)

        # Save the results in `html` file
        scan_results_path = f"reports/validation_results_{model_name}_{model_version}_{dataset_name}_{version}.html"
        scan_results.to_html(scan_results_path)

        suite_name = f"test_suite_{model_name}_{model_version}_{dataset_name}_{version}"
        test_suite = giskard.Suite(name = suite_name)

        test1 = giskard.testing.test_mae(model = giskard_model, 
                                    dataset = giskard_dataset,
                                    threshold=cfg.threshold.mae)
        
        test2 = giskard.testing.test_r2(model = giskard_model, 
                                    dataset = giskard_dataset,
                                    threshold=cfg.threshold.r2)
        

        test_suite.add_test(test1)
        test_suite.add_test(test2)

        test_results = test_suite.run()
        if (test_results.passed):
            print("Passed model validation!")
        else:
            print("Model has vulnerabilities!")