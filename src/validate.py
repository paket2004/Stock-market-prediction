from data import extract_data, preprocess_data # custom module
from model import retrieve_model_with_alias # custom module
import giskard
import hydra
from hydra import initialize, compose
import os
import yaml
import mlflow


def predict(raw_df, model):
    
    # Transform raw data into features
    X, _ = preprocess_data(df=raw_df)
    
    # Predict using the model
    predictions = model.predict(X)
    
    return predictions



current_dir = os.path.dirname(__file__)

file_in_another_directory = os.path.join(current_dir, '..', 'configs')
relative_path = os.path.relpath(file_in_another_directory, current_dir)

config_name="config"

with initialize(config_path=relative_path):
    cfg = compose(config_name=config_name)

    counter_file_path = os.path.join('..', cfg.batch.counter_file)


    with open(counter_file_path, 'r') as counter_file:
        counter_data = yaml.safe_load(counter_file)
        version = counter_data.get('file_version', 0)



    df = extract_data(str(version+1))

    # Specify categorical columns and target column
    TARGET_COLUMN = cfg.data.target_cols[0]

    CATEGORICAL_COLUMNS = list(cfg.data.cat_cols) + list(cfg.data.bin_cols)

    dataset_name = cfg.dataset.name


    # Wrap your Pandas DataFrame with giskard.Dataset (validation or test set)
    giskard_dataset = giskard.Dataset(
        df=df,  # A pandas.DataFrame containing raw data (before pre-processing) and including ground truth variable.
        target=TARGET_COLUMN,  # Ground truth variable
        name=dataset_name, # Optional: Give a name to your dataset
        cat_columns=CATEGORICAL_COLUMNS  # List of categorical columns. Optional, but improves quality of results if available.
    )

    model_name = cfg.best_model_name

# You can sweep over challenger aliases using Hydra
    model_alias = cfg.model.best_model_alias

    model, model_version = retrieve_model_with_alias(model_name, model_alias = model_alias)  


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

    test1 = giskard.testing.test_f1(model = giskard_model, 
                                dataset = giskard_dataset,
                                threshold=cfg.model.f1_threshold)

    test_suite.add_test(test1)

    test_results = test_suite.run()
    if (test_results.passed):
        print("Passed model validation!")
    else:
        print("Model has vulnerabilities!")