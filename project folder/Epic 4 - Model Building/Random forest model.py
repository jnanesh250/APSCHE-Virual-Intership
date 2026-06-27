import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import argparse
from pathlib import Path
import joblib


def random_forest(X_train, X_test, y_train, y_test, n_estimators=100, random_state=42, save_model_path=None):
    """Trains, evaluates, and optionally saves a Random Forest classifier.

    Returns the trained model and predicted labels for X_test.
    """
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    print("\nTraining Random Forest model...")
    rf_model.fit(X_train, y_train)

    print("Generating predictions...")
    y_pred = rf_model.predict(X_test)

    print("\n" + "=" * 50)
    print("Random Forest Model Evaluation")
    print("=" * 50)

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred, zero_division=0))

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    if save_model_path:
        try:
            joblib.dump(rf_model, save_model_path)
            print(f"Saved Random Forest model to {save_model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    return rf_model, y_pred


def find_dataset_by_name(name='creditcard.csv'):
    start = Path(__file__).resolve().parent
    for base in [start] + list(start.parents):
        try:
            for match in base.rglob(name):
                return match
        except Exception:
            continue
    return None


def parse_args():
    p = argparse.ArgumentParser(description='Train and evaluate a Random Forest model')
    p.add_argument('--data', '-d', help='Path to CSV dataset', default=None)
    p.add_argument('--target', '-t', help='Target column name (if omitted, last column is used)', default=None)
    p.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    p.add_argument('--n-estimators', type=int, default=100, help='Number of trees')
    p.add_argument('--save-model', type=str, default=None, help='Path to save trained model (joblib)')
    p.add_argument('--save-predictions', type=str, default=None, help='Path to save predictions CSV')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    df = None
    if args.data:
        path = Path(args.data)
        if path.is_file():
            df = pd.read_csv(path)
        else:
            print(f"Data file not found at: {path}")

    if df is None:
        candidate = find_dataset_by_name('creditcard.csv')
        if candidate:
            try:
                df = pd.read_csv(candidate)
                print(f"Loaded dataset from {candidate}")
            except Exception as e:
                print(f"Found {candidate} but failed to read it: {e}")

    if df is None:
        print("Could not find dataset. Provide --data or place 'creditcard.csv' in the repo.")
    else:
        target = args.target if args.target else ('Class' if 'Class' in df.columns else df.columns[-1])
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y if len(y.unique())>1 else None)

        model, y_pred = random_forest(X_train, X_test, y_train, y_test, n_estimators=args.n_estimators, save_model_path=args.save_model)

        if args.save_predictions:
            try:
                out = X_test.copy()
                out['y_test'] = y_test.values
                out['y_pred'] = y_pred
                out.to_csv(args.save_predictions, index=False)
                print(f"Saved predictions to {args.save_predictions}")
            except Exception as e:
                print(f"Failed to save predictions: {e}")
