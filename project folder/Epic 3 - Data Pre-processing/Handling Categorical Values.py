import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "Epic 1 - Data Collection",
    "creditcard.csv"
)

CATEGORICAL_COLUMNS = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
]


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    encoders = {}

    for column in CATEGORICAL_COLUMNS:
        if column not in df.columns:
            raise KeyError(f"Missing expected categorical column: {column}")

        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))
        encoders[column] = encoder
        print(f"Encoded {column}: {list(encoder.classes_)}")

    return df, encoders


def save_dataset(df: pd.DataFrame, path: str) -> None:
    output_path = os.path.splitext(path)[0] + "_categorical_encoded.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved encoded dataset to: {output_path}")


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    encoded_df, _ = encode_categorical_columns(df)
    save_dataset(encoded_df, DATA_PATH)
