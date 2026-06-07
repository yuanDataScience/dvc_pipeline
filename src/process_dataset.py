import pandas as pd

from utils import (
    DROP_COLUMNS,
    TARGET_ENCODE_COLUMNS,
    ONE_HOT_COLUMNS,
    NUMERICAL_COLUMNS,
    TRAINING_DATASET,
    TARGET_COLUMN,
    FILL_COLUMNS,
    PROCESSED_DATASET,
    SEED,
    load_data
)


def read_data(file_name: str) -> pd.DataFrame:
    return pd.read_csv(file_name)


def drop_columns(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    return df.drop(columns=drop_cols)


def convert_column_values(df: pd.DataFrame, col: str, convert_map: dict) -> pd.DataFrame:
    df[col] = df[col].map(convert_map)
    return df


def fill_missing_values(df: pd.DataFrame, cols: list[str], replace_value: str) -> pd.DataFrame:
    df[cols] = df[cols].fillna(replace_value)

    return df


def main():
    """
    Reads the raw data file, processes data and saves the processed data
    1. read training dataset from TRAINING_DATASET
    2. drop columns defined in DROP_COLUMNS
    3. convert target column to binary values
    4. fill missing values of FILL_COLUMNS using 'other'
    5. save processed data in PROCESSED_DATASET
    Target column values are expected in binary format with Yes/No values
    """

    df = read_data(TRAINING_DATASET)
    df = drop_columns(df, DROP_COLUMNS)

    df = convert_column_values(df, TARGET_COLUMN, {'<=50K': 0, '>50K': 1})
    df = fill_missing_values(df, FILL_COLUMNS, 'other')

    df.to_csv(PROCESSED_DATASET, index=False)


if __name__ == "__main__":
    main()

