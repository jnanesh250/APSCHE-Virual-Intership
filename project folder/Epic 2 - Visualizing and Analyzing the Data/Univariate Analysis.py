"""
Univariate Analysis utilities for Epic 2: Visualizing and Analyzing the Data

This script provides functions and example code to perform univariate analysis
on categorical and numerical features using pandas and seaborn/matplotlib.

Usage:
    python "Univariate Analysis.py" --app path/to/application_record.csv

Outputs:
    - prints value counts for selected categorical columns
    - saves example plots (countplots, histograms) to the local folder
"""

import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_count(feature_series: pd.Series, title: str, output_path: str = None):
    plt.figure(figsize=(12, 6))
    sns.countplot(y=feature_series, order=feature_series.value_counts().index, palette='Set2')
    plt.title(title)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()


def analyze_categorical(df: pd.DataFrame, col: str, save_plots: bool = False):
    print(f"\nValue counts for `{col}`:")
    counts = df[col].value_counts(dropna=False)
    print(counts)
    if save_plots:
        out_file = f"{col}_countplot.png"
    else:
        out_file = None
    plot_count(df[col].fillna('MISSING'), f"Distribution of {col}", out_file)


def analyze_numeric(df: pd.DataFrame, col: str, save_plots: bool = False):
    print(f"\nDescribe `{col}`:")
    print(df[col].describe())
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{col}_hist.png")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Run univariate analysis on dataset')
    parser.add_argument('--app', help='Path to application_record.csv', default='application_record.csv')
    parser.add_argument('--col', help='Column to analyze', default='OCCUPATION_TYPE')
    parser.add_argument('--save', action='store_true', help='Save plots to files')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.app):
        print(f"File not found: {args.app}")
    else:
        df = pd.read_csv(args.app)
        if args.col in df.columns:
            if df[args.col].dtype == 'O' or df[args.col].dtype.name == 'category':
                analyze_categorical(df, args.col, save_plots=args.save)
            else:
                analyze_numeric(df, args.col, save_plots=args.save)
        else:
            print(f"Column {args.col} not found in {args.app}")
