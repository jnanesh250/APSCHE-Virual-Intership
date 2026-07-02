import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic credit approval dataset with ONLY the required fields
n_samples = 5000

data = {
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Own_Car': np.random.choice(['Yes', 'No'], n_samples),
    'Own_Real_Estate': np.random.choice(['Yes', 'No'], n_samples),
    'Income_Type': np.random.choice(['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'], n_samples),
    'Education': np.random.choice(['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'], n_samples),
    'Family_Status': np.random.choice(['Single / not married', 'Married', 'Civil marriage', 'Separated', 'Widow'], n_samples),
    'Housing_Type': np.random.choice(['House / apartment', 'Rented apartment', 'Municipal apartment', 'With parents', 'Co-op apartment', 'Office apartment'], n_samples),
    'Annual_Income': np.random.uniform(10000, 300000, n_samples),
    'Days_Birth': np.random.uniform(-25000, -7000, n_samples),  # Negative days (age proxy)
    'Days_Employed': np.random.uniform(-5000, 10000, n_samples),
    'Family_Members': np.random.randint(1, 6, n_samples),
    'EMI_Paid_Off': np.random.uniform(0, 50000, n_samples),
    'EMI_Past_Dues': np.random.uniform(0, 30000, n_samples),
    'Number_of_Loans': np.random.randint(0, 10, n_samples),
}

df = pd.DataFrame(data)

# Create approval target based on some logical rules
# Higher income, more family members, employed, fewer past dues = higher approval probability
df['Approval'] = (
    (df['Annual_Income'] > 50000) & 
    (df['Days_Employed'] > -2000) & 
    (df['EMI_Past_Dues'] < 10000) &
    (df['Number_of_Loans'] < 5)
).astype(int)

# Add some randomness to make it more realistic (20% random flip)
random_flip = np.random.rand(n_samples) < 0.2
df.loc[random_flip, 'Approval'] = 1 - df.loc[random_flip, 'Approval']

print("Dataset Created!")
print(f"Shape: {df.shape}")
print(f"\nApproval Distribution:")
print(df['Approval'].value_counts())
print(f"\nFirst few rows:")
print(df.head())

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Own_Car', 'Own_Real_Estate', 'Income_Type', 'Education', 'Family_Status', 'Housing_Type']

df_encoded = df.copy()
for col in categorical_columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features and target
X = df_encoded.drop('Approval', axis=1)
y = df_encoded['Approval']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train Random Forest model with only the required features
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=10)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print("RANDOM FOREST MODEL PERFORMANCE")
print("="*60)
print(f"Accuracy: {accuracy:.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

# Feature importance
print(f"\nFeature Importance:")
feature_names = X.columns.tolist()
for fname, importance in sorted(zip(feature_names, rf_model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {fname}: {importance:.4f}")

# Save the model
joblib.dump(rf_model, 'rf_model.joblib')
print("\n✓ Model saved as rf_model.joblib")

# Save label encoders for later use during prediction
joblib.dump(label_encoders, 'label_encoders.joblib')
print("✓ Label encoders saved as label_encoders.joblib")

# Save feature names
joblib.dump(feature_names, 'feature_names.joblib')
print("✓ Feature names saved as feature_names.joblib")
