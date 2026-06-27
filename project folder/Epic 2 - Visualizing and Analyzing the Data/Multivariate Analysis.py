"""
Multivariate Analysis for Epic 2 - Visualizing and Analyzing the Data

This script computes a correlation matrix for numeric features and
visualizes it as a heatmap to help identify relationships and
potential multicollinearity among features.

Usage:
    python "Multivariate Analysis.py" --app path/to/application_record.csv --save

Options:
    --app   Path to the application CSV file (default: application_record.csv)
    --save  If provided, saves the heatmap image to `correlation_heatmap.png`
"""

import argparse
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def compute_and_plot_correlations(df: pd.DataFrame, save: bool = False, out_file: str = 'correlation_heatmap.png') -> None:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        print('No numeric columns found for correlation analysis.')
        return

    corr = numeric.corr()
    print('\nCorrelation matrix (top-left 10x10 preview):')
    print(corr.iloc[:10, :10])

    plt.figure(figsize=(12, 10))
    sns.set(style='white')
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()

    if save:
        plt.savefig(out_file, dpi=300)
        print(f'Saved heatmap to {out_file}')
    plt.show()


def list_high_correlations(corr: pd.DataFrame, threshold: float = 0.7):
    # list pairs with absolute correlation above threshold (excluding self-pairs)
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append((cols[i], cols[j], val))
    if pairs:
        print('\nHigh-correlation pairs (|corr| >=', threshold, '):')
        for a, b, v in sorted(pairs, key=lambda x: -abs(x[2])):
            print(f'  {a} <-> {b}: {v:.3f}')
    else:
        print(f'\nNo pairs found with absolute correlation >= {threshold}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run multivariate correlation analysis and heatmap')
    parser.add_argument('--app', help='Path to application_record.csv', default='application_record.csv')
    parser.add_argument('--save', action='store_true', help='Save the heatmap image to file')
    parser.add_argument('--threshold', type=float, default=0.7, help='Threshold for reporting high correlations')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.app):
        print(f'File not found: {args.app}')
    else:
        df = pd.read_csv(args.app)
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            print('No numeric columns found in dataset. Consider converting appropriate columns to numeric types.')
        else:
            corr = numeric.corr()
            compute_and_plot_correlations(df, save=args.save)
            list_high_correlations(corr, threshold=args.threshold)
