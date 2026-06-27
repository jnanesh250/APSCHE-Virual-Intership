import os
import pandas as pd

BASE_DIR = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "Epic 1 - Data Collection"
)

APPLICANT_FILE = os.path.join(BASE_DIR, "creditcard.csv")
CREDIT_RECORD_FILE = os.path.join(BASE_DIR, "credit_record.csv")

CATEGORY_MAPS = {
    "NAME_HOUSING_TYPE": {
        "House / apartment": 0,
        "Municipal apartment": 1,
        "Rented apartment": 2,
        "Office apartment": 3,
        "Co-op apartment": 4,
    },
    "NAME_INCOME_TYPE": {
        "Working": 0,
        "State servant": 1,
        "Commercial associate": 2,
        "Pensioner": 3,
        "Student": 4,
    },
    "NAME_EDUCATION_TYPE": {
        "Secondary / secondary special": 0,
        "Higher education": 1,
        "Incomplete higher": 2,
        "Lower secondary": 3,
        "Academic degree": 4,
    },
    "NAME_FAMILY_STATUS": {
        "Single / not married": 0,
        "Married": 1,
        "Civil marriage": 2,
        "Separated": 3,
        "Widow": 4,
    },
    "FLAG_OWN_CAR": {"Y": 1, "N": 0},
    "FLAG_OWN_REALTY": {"Y": 1, "N": 0},
}


def load_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    return pd.read_csv(file_path)


def map_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column, mapping in CATEGORY_MAPS.items():
        if column in df.columns:
            df[column + "_ENCODED"] = df[column].map(mapping).fillna(-1).astype(int)
    return df


def clean_applicant_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "CNT_FAM_MEMBERS" in df.columns and "CNT_CHILDREN" in df.columns:
        df["FAMILY_DEPENDENCY"] = df["CNT_FAM_MEMBERS"] + df["CNT_CHILDREN"]

    if "DAYS_BIRTH" in df.columns:
        df["DAYS_BIRTH"] = df["DAYS_BIRTH"].abs()
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].abs()

    df = map_categorical_columns(df)

    drop_columns = [
        "OCCUPATION_TYPE",
        "NAME_HOUSING_TYPE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
    ]
    existing_drops = [col for col in drop_columns if col in df.columns]
    df = df.drop(columns=existing_drops)

    return df


def summarize_credit_records(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "MONTHS_BALANCE" not in df.columns or "ID" not in df.columns:
        raise KeyError("Credit record dataset must contain 'ID' and 'MONTHS_BALANCE' columns.")

    df["MONTHS_BALANCE"] = pd.to_numeric(df["MONTHS_BALANCE"], errors="coerce")
    df["STATUS_STR"] = df["STATUS"].astype(str)

    status_features = df.groupby("ID").apply(
        lambda group: pd.Series({
            "open_month": group["MONTHS_BALANCE"].min(),
            "end_month": group["MONTHS_BALANCE"].max(),
            "window": group["MONTHS_BALANCE"].max() - group["MONTHS_BALANCE"].min(),
            "on_time_payments": (group["STATUS_STR"] == "0").sum(),
            "overdue_payments": group["STATUS_STR"].isin(["1", "2", "3", "4", "5"]).sum(),
            "no_loan_records": group["STATUS_STR"].isin(["X", "C"]).sum(),
            "most_recent_status": group.loc[group["MONTHS_BALANCE"].idxmax(), "STATUS_STR"],
        })
    )

    status_features["open_month"] = status_features["open_month"].abs()
    status_features["end_month"] = status_features["end_month"].abs()

    return status_features.reset_index()


def merge_cleaned_data(applicant_df: pd.DataFrame, credit_df: pd.DataFrame) -> pd.DataFrame:
    if "ID" not in applicant_df.columns or "ID" not in credit_df.columns:
        raise KeyError("Both datasets must include an 'ID' column for merging.")

    return applicant_df.merge(credit_df, on="ID", how="left")


def save_dataset(df: pd.DataFrame, file_path: str) -> None:
    output_path = os.path.splitext(file_path)[0] + "_cleaned.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")


if __name__ == "__main__":
    applicant_df = load_dataset(APPLICANT_FILE)
    cleaned_applicant = clean_applicant_data(applicant_df)
    save_dataset(cleaned_applicant, APPLICANT_FILE)

    if os.path.exists(CREDIT_RECORD_FILE):
        credit_df = load_dataset(CREDIT_RECORD_FILE)
        credit_summary = summarize_credit_records(credit_df)
        save_dataset(credit_summary, CREDIT_RECORD_FILE)

        merged_df = merge_cleaned_data(cleaned_applicant, credit_summary)
        merged_output = os.path.join(BASE_DIR, "creditcard_merged.csv")
        merged_df.to_csv(merged_output, index=False)
        print(f"Merged applicant and credit data saved to: {merged_output}")
    else:
        print(f"Credit record file not found at: {CREDIT_RECORD_FILE}. Skipping merge.")
