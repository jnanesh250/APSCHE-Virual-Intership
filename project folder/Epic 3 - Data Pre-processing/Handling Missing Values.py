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


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Report missing value counts and proportions for each column."""
    missing_count = df.isnull().sum()
    missing_ratio = df.isnull().mean()
    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_ratio": missing_ratio,
    })
    print("Missing value report:\n", report)
    return report


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values by removing unneeded columns and confirming no nulls remain."""
    report = report_missing_values(df)
    if "OCCUPATION_TYPE" in df.columns:
        print("\nOCCUPATION_TYPE missing values detected and will be removed as part of preprocessing.")
        df = df.drop(columns=["OCCUPATION_TYPE"])
    else:
        print("\nOCCUPATION_TYPE is not present in the dataset or has already been removed.")

    final_missing = df.isnull().sum().sum()
    if final_missing == 0:
        print("\nNo missing values remain after processing.")
    else:
        print(f"\nWarning: {final_missing} missing values still remain after processing.")
        print(df.isnull().sum()[df.isnull().sum() > 0])

    return df


def save_cleaned_dataset(df: pd.DataFrame, path: str) -> None:
    """Save the cleaned dataset to a new CSV file."""
    output_path = os.path.splitext(path)[0] + "_missing_handled.csv"
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    cleaned_df = handle_missing_values(df)
    save_cleaned_dataset(cleaned_df, DATA_PATH)
