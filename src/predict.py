import json
import requests
import hydra
import mlflow
from data import extract_data

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def predict(cfg = None):

    data = extract_data(version = cfg.example_version)
    X = data.drop(['Adj Close'], axis=1)
    y = data['Adj Close']

    # 1 sample example
    example = X.iloc[0,:]
    example_target = y[0]

    example = json.dumps( 
    { "inputs": example.to_dict() }
    )

    payload = example

    response = requests.post(
        url=f"http://localhost:{cfg.port}/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    print(response.json())
    print("encoded target labels: ", example_target)
    print("target labels: ", list(cfg.data.labels)[example_target])


if __name__=="__main__":
    predict()