# Stock-market-prediction
This repository contains code for predicting stock market trends using various machine learning and data processing techniques. The main goal of this project is to analyze historical stock market data and create models that can predict future stock prices.

![Test code workflow](https://github.com/paket2004/Stock-Market-Prediction/actions/workflows/test-code.yml/badge.svg)

## Installation
To set up the project, follow these steps:
  1. Clone the repository:
  ```git clone https://github.com/paket2004/Stock-market-prediction.git
     cd Stock-market-prediction
  ```
  2. Activate the virtual environment:
     windows:
     ```
     env\Scripts\activate
     ```
     Linux/macOS:
     ```
     source env/bin/activate
     ```
  3. Install the required dependencies:
     ```
     pip install -r requirements.txt
     ```
## Usage
To run the main script for downloading and processing data, use the following command:
```
python src/main.py
```
To run the tests, use pytest:
```
pytest tests/
```
## Configuration
The project uses Hydra for configuration management. The main configuration file is located at configs/config.yaml. You can modify this file to change various parameters such as dataset URLs, batch sizes, and paths.

Here is an example configuration:
```
batch:
  counter_file: "data/counter.txt"
  random_seed: 42
  size: 10
  save_dir: "data/samples"

dataset:
  url: "zillow/zecon"
  archive_name: "archive.zip"
  file_name: "file.csv"
```
## Project Structure
├───README.md          # Repo docs

├───.gitignore         # gitignore file

├───requirements.txt   # Python packages   

├───configs            # Hydra configuration management

├───data               # All data

├───docs               # Project docs like reports or figures

├───models             # ML models

├───notebooks          # Jupyter notebooks

├───outputs            # Outputs of Hydra

├───pipelines          # A Soft link to DAGs of Apache Airflow

├───reports            # Generated reports 

├───scripts            # Shell scripts (.sh)

├───services           # Metadata of services (PostgreSQL, Feast, Apache airflow, ...etc)

├───sql                # SQL files

├───src                # Python scripts
└───tests              # Scripts for testing Python code
