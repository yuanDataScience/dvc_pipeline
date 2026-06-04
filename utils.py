import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_openml 

DATASET_TYPES = ["test", "train"]
DROP_COLNAMES = ["Date", "Evaporation", "Sunshine", "RISK_MM"]
CATEGORICAL_COLUMNS = [
    'Location', 'WindGustDir', 'WindDir9am',
    'WindDir3pm', 'RainToday']
TARGET_COLUMN = "RainTomorrow"
RAW_DATASET = "raw_dataset/weather.csv"
PROCESSED_DATASET = "processed_dataset/weather.csv"
RFC_FOREST_DEPTH = 2
SEED = 42


def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)

def initialize_dataset():
           


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y


def load_hyperparameters(hyperparameter_file):
    with open(hyperparameter_file, "r") as json_file:
        hyperparameters = json.load(json_file)
    return hyperparameters


