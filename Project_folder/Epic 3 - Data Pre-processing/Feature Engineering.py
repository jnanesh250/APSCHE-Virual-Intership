import os
import pandas as pd

BASE_DIR = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "Epic 1 - Data Collection"
)

APPLICANT_FILE = os.path.join(BASE_DIR, "creditcard.csv")
CREDIT_RECORD_FILE = os.path.join(BASE_DIR, "credit_record.csv")


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def to_binary(status: str) -> int:
    """Convert multi-class payment status into binary approval labels."""
    status = str(status).strip()
    if status in {"0", "X", "C"}:
        return 1
    return 0


def add_status_binary_column(df: pd.DataFrame) -> pd.DataFrame:
    if "STATUS" not in df.columns:
        raise KeyError("Credit record dataset is missing the STATUS column.")
    df = df.copy()
    df["STATUS_BIN"] = df["STATUS"].apply(to_binary)
    print("STATUS_BIN value counts:\n", df["STATUS_BIN"].value_counts())
    return df


def aggregate_credit_history(df: pd.DataFrame) -> pd.DataFrame:
    if "ID" not in df.columns:
        raise KeyError("Credit record dataset is missing the ID column.")

    summary = (
        df.groupby("ID")
          .agg(
              total_records=("STATUS_BIN", "count"),
              approved_records=("STATUS_BIN", "sum"),
              not_approved_records=("STATUS_BIN", lambda x: (x == 0).sum()),
              most_recent_status=("STATUS", lambda x: x.iloc[-1]),
              final_status=("STATUS_BIN", lambda x: 1 if x.min() == 1 else 0),
          )
          .reset_index()
    )
    print("Credit history aggregated by ID. Sample rows:\n", summary.head())
    return summary


def merge_with_applicant_data(applicant_df: pd.DataFrame, credit_summary: pd.DataFrame) -> pd.DataFrame:
    if "ID" not in applicant_df.columns:
        raise KeyError("Applicant dataset is missing the ID column.")
    merged_df = applicant_df.merge(credit_summary, on="ID", how="left")
    print(f"Merged applicant and credit summary shape: {merged_df.shape}")
    return merged_df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    output_path = os.path.splitext(path)[0] + "_feature_engineered.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved feature-engineered dataset to: {output_path}")


if __name__ == "__main__":
    applicant_df = load_dataset(APPLICANT_FILE)
    credit_df = load_dataset(CREDIT_RECORD_FILE)

    credit_df = add_status_binary_column(credit_df)
    credit_summary = aggregate_credit_history(credit_df)

    final_df = merge_with_applicant_data(applicant_df, credit_summary)
    save_dataset(final_df, os.path.join(BASE_DIR, "creditcard_final.csv"))
