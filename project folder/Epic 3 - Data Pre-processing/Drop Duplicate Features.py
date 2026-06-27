import os
import pandas as pd

# Update this path if the dataset is stored elsewhere in the workspace.
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "Epic 1 - Data Collection",
    "creditcard.csv"
)


def load_dataset(path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def drop_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate records using a subset of relevant applicant features."""
    duplicate_subset = [
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "CNT_CHILDREN",
        "AMT_INCOME_TOTAL",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "FLAG_MOBIL",
        "FLAG_WORK_PHONE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "OCCUPATION_TYPE",
        "CNT_FAM_MEMBERS",
    ]

    missing_columns = [col for col in duplicate_subset if col not in df.columns]
    if missing_columns:
        raise KeyError(
            f"The dataset is missing required columns for duplicate removal: {missing_columns}"
        )

    before_count = len(df)
    cleaned_df = df.drop_duplicates(subset=duplicate_subset, keep="first")
    after_count = len(cleaned_df)

    print(f"Original records: {before_count}")
    print(f"Records after removing duplicates: {after_count}")
    print(f"Duplicates removed: {before_count - after_count}")

    return cleaned_df


def save_cleaned_dataset(df: pd.DataFrame, path: str) -> None:
    """Save the cleaned dataset to a new CSV file."""
    output_path = os.path.splitext(path)[0] + "_cleaned.csv"
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    cleaned_df = drop_duplicate_records(df)
    save_cleaned_dataset(cleaned_df, DATA_PATH)