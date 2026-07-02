"""
Descriptive Analysis for Epic 2 - Visualizing and Analyzing the Data

This script computes descriptive statistics for numerical (and optional categorical)
features using pandas' `describe()` and prints key measures: count, mean, std,
min, percentiles (25%, 50%, 75%), and max. It also reports missing values and
can optionally save the summary to CSV.

Usage:
    python "descriptive Analysis.py" --app path/to/application_record.csv [--save-summary summary.csv]

Options:
    --app            Path to application_record.csv (default: application_record.csv)
    --save-summary   Optional path to save descriptive statistics CSV
    --include-all    Include all columns (numeric and categorical) in describe()
"""

import argparse
import os
import pandas as pd


def descriptive_summary(df: pd.DataFrame, include_all: bool = False) -> pd.DataFrame:
    if include_all:
        desc = df.describe(include='all')
    else:
        desc = df.describe()
    return desc


def print_summary(desc: pd.DataFrame) -> None:
    print('\nDescriptive statistics:')
    print(desc)


def print_missing(df: pd.DataFrame) -> None:
    print('\nMissing values per column:')
    print(df.isnull().sum())


def save_summary(desc: pd.DataFrame, path: str) -> None:
    try:
        desc.to_csv(path)
        print(f'Saved descriptive summary to {path}')
    except Exception as e:
        print('Error saving summary:', e)


def parse_args():
    parser = argparse.ArgumentParser(description='Run descriptive analysis on dataset')
    parser.add_argument('--app', help='Path to application_record.csv', default='application_record.csv')
    parser.add_argument('--save-summary', help='Path to save descriptive summary CSV', default=None)
    parser.add_argument('--include-all', action='store_true', help='Include all columns in describe()')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.app):
        print(f'File not found: {args.app}')
    else:
        df = pd.read_csv(args.app)
        print(f"Loaded dataset: {args.app} (shape: {df.shape})")

        # Descriptive statistics
        desc = descriptive_summary(df, include_all=args.include_all)
        print_summary(desc)

        # Missing values
        print_missing(df)

        # Optionally save
        if args.save_summary:
            save_summary(desc, args.save_summary)
