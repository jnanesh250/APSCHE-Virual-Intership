"""
Data inspection utility for Epic 2 - Visualizing and Analyzing the Data

This script demonstrates reading CSV files with pandas and running
basic exploratory inspection methods (`head()`, `shape`, `info()`,
`isnull().sum()`, `describe()`) to understand structure, features,
and available records prior to preprocessing and modelling.

Usage:
    python data_inspection.py --app path/to/application_record.csv --credit path/to/credit_record.csv

If a path is not provided or the file is missing, the script will report it.
"""

import argparse
import os
import pandas as pd


def inspect_dataframe(df: pd.DataFrame, name: str) -> None:
    print('\n' + '=' * 80)
    print(f'Inspecting: {name}')
    print('=' * 80)
    print('Shape:', df.shape)
    print('\nColumns:')
    print(list(df.columns))
    print('\nData types:')
    print(df.dtypes)
    print('\nTop rows:')
    print(df.head(5))
    print('\nInfo:')
    df.info()
    print('\nMissing values (per column):')
    print(df.isnull().sum())
    print('\nDescriptive statistics (numeric):')
    print(df.describe())
    print('\nDescriptive statistics (all):')
    print(df.describe(include="all"))


def read_and_inspect(path: str, name: str) -> None:
    if not path:
        print(f'No path provided for {name}.')
        return
    if not os.path.exists(path):
        print(f'File not found: {path}')
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f'Error reading {path}:', e)
        return

    inspect_dataframe(df, name)


def parse_args():
    parser = argparse.ArgumentParser(description='Inspect application and credit CSV files')
    parser.add_argument('--app', help='Path to application_record.csv', default='application_record.csv')
    parser.add_argument('--credit', help='Path to credit_record.csv', default='credit_record.csv')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    read_and_inspect(args.app, 'Application Records')
    read_and_inspect(args.credit, 'Credit Records')
