import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

DATASET_TYPES = ["test", "train"]
DROP_COLUMNS = ["education"]
FILL_COLUMNS = ["workclass"]
TARGET_ENCODE_COLUMNS = ["workclass", "marital-status", "occupation",
                         "relationship", "native-country"]
ONE_HOT_COLUMNS = ["race", "sex"]
NUMERICAL_COLUMNS = ["age", "fnlwgt", "education-num", "capital-gain",
                     "capital-loss", "hours-per-week"]
TARGET_COLUMN = "class"
TRAINING_DATASET = "raw_dataset/train.csv"
TESTING_DATASET = "raw_dataset/test.csv"
PROCESSED_DATASET = "processed_dataset/train.csv"
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
    adult = fetch_openml(
        "adult",
        version=2,
        as_frame=True
    ).frame

    df_train, df_test = train_test_split(
        adult,
        test_size=0.2,
        stratify=adult["class"],
        random_state=SEED
    )

    df_train.to_csv(TRAINING_DATASET, index=False)
    df_test.to_csv(TESTING_DATASET, index=False)

def load_data(filename: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Reads the raw data file and returns pandas dataframe
    Target column values are expected in binary format with 0/1 values

    Parameters:
    filename (str): raw data filename
    drop_columns (List[str]): column names that will be dropped
    target_column (str): name of target column

    Returns:
    pd.Dataframe: Target encoded dataframe
    """
    data = pd.read_csv(filename)
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y


def load_hyperparameters(hyperparameter_file):
    with open(hyperparameter_file, "r") as json_file:
        hyperparameters = json.load(json_file)
    return hyperparameters

if __name__ == "__main__":
    initialize_dataset()
