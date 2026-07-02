import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning
import joblib
try:
    import seaborn as sns
except Exception:
    sns = None


def build_and_evaluate_logistic_regression(
    df,
    target_column,
    test_size=0.2,
    random_state=42,
    scale=True,
    max_iter=2000,
    solver='lbfgs',
    class_weight=None,
    save_model_path=None,
):
    """Train and evaluate a Logistic Regression model.

    Args:
        df (pd.DataFrame): Dataset containing features and target.
        target_column (str): Name of the target column in `df`.
        test_size (float): Fraction of data to reserve for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        model: Trained LogisticRegression model.
        results (dict): Dictionary with predictions and evaluation metrics.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique())>1 else None
    )

    clf_kwargs = dict(max_iter=max_iter, random_state=random_state, solver=solver)
    if class_weight is not None:
        clf_kwargs['class_weight'] = class_weight

    base_clf = LogisticRegression(**clf_kwargs)

    if scale:
        model = Pipeline([('scaler', StandardScaler()), ('clf', base_clf)])
    else:
        model = base_clf

    # Fit and catch convergence warnings so user can see suggestions
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X_train, y_train)
        for wi in w:
            if issubclass(wi.category, ConvergenceWarning):
                print('Warning during training:', wi.message)

    if save_model_path:
        try:
            joblib.dump(model, save_model_path)
            print(f"Saved trained model to {save_model_path}")
        except Exception as e:
            print(f"Failed to save model to {save_model_path}: {e}")

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    results = {
        "confusion_matrix": cm,
        "classification_report": report,
        "accuracy": acc,
        "y_test": y_test,
        "y_pred": y_pred,
        "X_test": X_test,
    }

    return model, results


def plot_confusion_matrix(cm, labels=None, figsize=(6, 5)):
    plt.figure(figsize=figsize)
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=labels, yticklabels=labels)
    else:
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, int(val), ha='center', va='center', color='black')
        if labels is not None:
            plt.xticks(range(len(labels)), labels)
            plt.yticks(range(len(labels)), labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix - Logistic Regression')
    plt.tight_layout()
    plt.show()


def find_dataset_by_name(name='creditcard.csv'):
    """Search upward from the script directory for the given filename."""
    start = Path(__file__).resolve().parent
    # search the start directory and all its parents
    for base in [start] + list(start.parents):
        try:
            for match in base.rglob(name):
                return match
        except Exception:
            continue
    return None


def parse_args():
    p = argparse.ArgumentParser(description='Train and evaluate a Logistic Regression model')
    p.add_argument('--data', '-d', help='Path to CSV dataset', default=None)
    p.add_argument('--target', '-t', help='Target column name (if omitted, last column is used)', default=None)
    p.add_argument('--no-scale', dest='scale', action='store_false', help='Disable feature scaling')
    p.add_argument('--max-iter', type=int, default=2000, help='Max iterations for LogisticRegression')
    p.add_argument('--solver', type=str, default='lbfgs', help='Solver for LogisticRegression')
    p.add_argument('--class-weight', type=str, default=None, help="Class weight ('balanced' or omit)")
    p.add_argument('--save-model', type=str, default=None, help='Path to save trained model (joblib)')
    p.add_argument('--save-predictions', type=str, default=None, help='Path to save predictions CSV')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    df = None
    if args.data:
        data_path = Path(args.data)
        if not data_path.is_file():
            print(f"Data file not found at: {data_path}")
        else:
            df = pd.read_csv(data_path)

    if df is None:
        # try common default name first
        candidate = find_dataset_by_name('creditcard.csv')
        if candidate is not None:
            try:
                df = pd.read_csv(candidate)
                print(f"Loaded dataset from {candidate}")
            except Exception as e:
                print(f"Found {candidate} but failed to read it: {e}")

    if df is None:
        # try any CSV in the repository root or 'Epic 1 - Data Collection' folder
        parents = Path(__file__).resolve().parents
        repo_root = parents[2] if len(parents) >= 3 else Path.cwd()
        possible = list(repo_root.rglob('*.csv'))
        if possible:
            print("Found CSV files in repository. Try running with --data pointing to one of:")
            for pth in possible[:20]:
                print(' -', pth)
        print("Could not read dataset automatically. Please pass --data /path/to/file.csv or place 'creditcard.csv' in the repo.")
    else:
        target = args.target if args.target else ('Class' if 'Class' in df.columns else df.columns[-1])
        class_weight = None if not args.class_weight else args.class_weight
        model, res = build_and_evaluate_logistic_regression(
            df,
            target,
            scale=args.scale,
            max_iter=args.max_iter,
            solver=args.solver,
            class_weight=class_weight,
            save_model_path=args.save_model,
        )
        # Save predictions and reports if requested
        if args.save_predictions:
            try:
                preds_df = res['X_test'].copy()
                preds_df['y_test'] = res['y_test'].values
                preds_df['y_pred'] = res['y_pred']
                # try to add probabilities if available
                try:
                    probs = model.predict_proba(res['X_test'])
                    # if binary, keep prob for class 1
                    if probs.shape[1] == 2:
                        preds_df['prob_class_1'] = probs[:, 1]
                    else:
                        for i in range(probs.shape[1]):
                            preds_df[f'prob_class_{i}'] = probs[:, i]
                except Exception:
                    pass
                preds_df.to_csv(args.save_predictions, index=False)
                print(f"Saved predictions to {args.save_predictions}")
            except Exception as e:
                print(f"Failed to save predictions: {e}")
            # save classification report and confusion matrix
            try:
                report_path = Path(args.save_predictions).with_suffix('.report.txt')
                with open(report_path, 'w') as fh:
                    fh.write(res['classification_report'])
                cm_path = Path(args.save_predictions).with_suffix('.cm.csv')
                pd.DataFrame(res['confusion_matrix']).to_csv(cm_path, index=False)
                print(f"Saved report to {report_path} and confusion matrix to {cm_path}")
            except Exception as e:
                print(f"Failed to save report/cm: {e}")
        print('Accuracy:', res['accuracy'])
        print('Classification Report:\n', res['classification_report'])
        labels = [0,1] if len(set(df[target]))==2 else None
        plot_confusion_matrix(res['confusion_matrix'], labels=labels)
