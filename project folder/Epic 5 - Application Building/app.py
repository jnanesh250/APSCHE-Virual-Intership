from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "rf_model.joblib")
ENCODERS_PATH = os.path.join(BASE_DIR, "model", "label_encoders.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_names.joblib")

# Load model and encoders
try:
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    feature_names = joblib.load(FEATURES_PATH)
except FileNotFoundError as e:
    model = None
    label_encoders = {}
    feature_names = []
    print(f"Error loading model files: {e}")

# Required input fields from the form
REQUIRED_FIELDS = [
    "gender", "own_car", "own_real_estate", "income_type", "education",
    "family_status", "housing_type", "annual_income", "days_birth",
    "days_employed", "family_members", "emi_paid_off", "emi_past_dues",
    "number_of_loans"
]

# Mapping from form field names to model feature names
FIELD_TO_FEATURE = {
    "gender": "Gender",
    "own_car": "Own_Car",
    "own_real_estate": "Own_Real_Estate",
    "income_type": "Income_Type",
    "education": "Education",
    "family_status": "Family_Status",
    "housing_type": "Housing_Type",
    "annual_income": "Annual_Income",
    "days_birth": "Days_Birth",
    "days_employed": "Days_Employed",
    "family_members": "Family_Members",
    "emi_paid_off": "EMI_Paid_Off",
    "emi_past_dues": "EMI_Past_Dues",
    "number_of_loans": "Number_of_Loans",
}


@app.route("/")
def home():
    """Render the landing page."""
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Render the form or process prediction requests."""
    if request.method == "POST":
        try:
            form_data = request.form
            
            # Check all required fields are present
            for field in REQUIRED_FIELDS:
                if field not in form_data or form_data[field] in ("", None):
                    raise ValueError(f"Missing required field: {field}")
            
            if model is None:
                raise FileNotFoundError("The trained model file is not available.")

            # Build feature vector by encoding categorical values and keeping numeric ones
            feature_vector = []
            feature_display = {}
            
            for field_name in REQUIRED_FIELDS:
                feature_name = FIELD_TO_FEATURE[field_name]
                value = form_data.get(field_name)
                
                # Check if this is a categorical field (needs encoding)
                if feature_name in label_encoders:
                    # Encode categorical value
                    encoder = label_encoders[feature_name]
                    try:
                        encoded_value = encoder.transform([value])[0]
                    except ValueError:
                        raise ValueError(f"Invalid value for {feature_name}: {value}")
                    feature_vector.append(encoded_value)
                    feature_display[feature_name] = value
                else:
                    # Keep numeric values as-is
                    try:
                        numeric_value = float(value)
                        feature_vector.append(numeric_value)
                        feature_display[feature_name] = numeric_value
                    except ValueError:
                        raise ValueError(f"Invalid numeric value for {feature_name}: {value}")
            
            # Create DataFrame with correct feature names
            df_input = pd.DataFrame([feature_vector], columns=feature_names)
            
            # Make prediction
            pred_raw = model.predict(df_input)
            prediction = int(pred_raw[0])
            
            # Get prediction probability
            try:
                proba = model.predict_proba(df_input)[0].tolist()
            except Exception:
                proba = None
            
            # Format result message
            prediction_message = "Credit Card Approved" if prediction == 1 else "Credit Card Rejected"
            
            # Create a clean display of input features (without V1, V2, etc.)
            feature_display_clean = {
                "Gender": feature_display.get("Gender", ""),
                "Income Type": feature_display.get("Income_Type", ""),
                "Education": feature_display.get("Education", ""),
                "Family Status": feature_display.get("Family_Status", ""),
                "Housing Type": feature_display.get("Housing_Type", ""),
                "Annual Income": f"${feature_display.get('Annual_Income', 0):,.2f}",
                "Days Employed": f"{feature_display.get('Days_Employed', 0):.0f}",
                "Family Members": f"{feature_display.get('Family_Members', 0):.0f}",
                "EMI Past Dues": f"${feature_display.get('EMI_Past_Dues', 0):,.2f}",
            }
            
            return render_template(
                "result.html",
                prediction=prediction_message,
                probability=proba,
                features=feature_display_clean
            )

        except ValueError as exc:
            return render_template("result.html", prediction=f"Input Error: {str(exc)}")
        except Exception as exc:
            return render_template("result.html", prediction=f"Prediction error: {str(exc)}")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
