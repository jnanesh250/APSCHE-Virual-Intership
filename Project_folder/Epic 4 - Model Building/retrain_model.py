import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv('../Epic 1 - Data Collection/creditcard.csv')

# Use only Time and Amount - the realistic features from the dataset
X = df[['Time', 'Amount']]
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest with these realistic features
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
print("Random Forest Model - Using Time & Amount")
print("=" * 50)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the retrained model
joblib.dump(rf_model, 'rf_model.joblib')
print("\n✓ Model saved as rf_model.joblib")
print(f"\nFeature importance:")
for feat, imp in zip(['Time', 'Amount'], rf_model.feature_importances_):
    print(f"  {feat}: {imp:.4f}")
