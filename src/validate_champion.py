from data import read_datastore, preprocess_data # custom module
from model import retrieve_model_with_alias # custom module
import giskard
from hydra import initialize, compose
import os
import yaml
import pickle


def predict(raw_df, model):
    X, _ = preprocess_data(df=raw_df)
    predictions = model.predict(X)
    return predictions


project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(__file__)

file_in_another_directory = os.path.join(current_dir, '..', 'configs')
relative_path = os.path.relpath(file_in_another_directory, current_dir)

config_name="config"

with initialize(config_path=relative_path):
    cfg = compose(config_name=config_name)

    counter_file_path = os.path.join(project_root_dir, cfg.batch.counter_file)

    with open(counter_file_path, 'r') as counter_file:
        counter_data = yaml.safe_load(counter_file)
        version = counter_data.get('file_version', 0)

    df = read_datastore(str(version+1))
    dataset_name = cfg.dataset.name

    # Specify categorical columns and target column
    TARGET_COLUMN = cfg.data.target_cols[0]
    CATEGORICAL_COLUMNS = list(cfg.data.cat_cols)

    # Wrap your Pandas DataFrame with giskard.Dataset (validation or test set)
    giskard_dataset = giskard.Dataset(
        df=df,  
        target=TARGET_COLUMN,
        name=dataset_name,
        cat_columns=CATEGORICAL_COLUMNS 
    )

    model_path = os.path.join(project_root_dir, 'api', 'model_dir', 'model.pkl')
    model = pickle.load(open(model_path,'rb'))

    def predict(raw_df):
        X, _ = preprocess_data(df=raw_df)
        predictions = model.predict(X)
        return predictions

    giskard_model = giskard.Model(
        model=predict,
        model_type="regression",
        feature_names=df.columns, 
    )

    scan_results = giskard.scan(giskard_model, giskard_dataset)

    suite_name = f"test_suite_champion_{dataset_name}_{version}"
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
