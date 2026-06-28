from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "rf_model.joblib")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

FEATURE_ORDER = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]


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

            required_fields = [
                "gender", "own_car", "own_real_estate", "income_type", "education",
                "family_status", "housing_type", "annual_income", "days_birth",
                "days_employed", "family_members", "emi_paid_off", "emi_past_dues",
                "number_of_loans"
            ]

            for field in required_fields:
                if field not in form_data or form_data[field] in ("", None):
                    raise ValueError(f"Missing required field: {field}")

            # Convert the form values into a feature vector expected by the saved model.
            # The original model was trained on numeric financial features, so the form
            # inputs are mapped into a simplified numeric representation for prediction.
            gender_code = 1.0 if form_data.get("gender") == "Female" else 0.0
            car_code = 1.0 if form_data.get("own_car") == "Yes" else 0.0
            realty_code = 1.0 if form_data.get("own_real_estate") == "Yes" else 0.0
            income_code = {
                "Working": 0.0,
                "Commercial associate": 1.0,
                "Pensioner": 2.0,
                "State servant": 3.0,
                "Student": 4.0,
            }.get(form_data.get("income_type"), 0.0)
            education_code = {
                "Secondary / secondary special": 0.0,
                "Higher education": 1.0,
                "Incomplete higher": 2.0,
                "Lower secondary": 3.0,
                "Academic degree": 4.0,
            }.get(form_data.get("education"), 0.0)
            family_code = {
                "Single / not married": 0.0,
                "Married": 1.0,
                "Civil marriage": 2.0,
                "Separated": 3.0,
                "Widow": 4.0,
            }.get(form_data.get("family_status"), 0.0)
            housing_code = {
                "House / apartment": 0.0,
                "Rented apartment": 1.0,
                "Municipal apartment": 2.0,
                "With parents": 3.0,
                "Co-op apartment": 4.0,
                "Office apartment": 5.0,
            }.get(form_data.get("housing_type"), 0.0)

            annual_income = float(form_data.get("annual_income", 0))
            days_birth = float(form_data.get("days_birth", 0))
            days_employed = float(form_data.get("days_employed", 0))
            family_members = float(form_data.get("family_members", 0))
            emi_paid_off = float(form_data.get("emi_paid_off", 0))
            emi_past_dues = float(form_data.get("emi_past_dues", 0))
            number_of_loans = float(form_data.get("number_of_loans", 0))

            feature_map = {
                "Time": annual_income / 1000.0,
                "V1": days_birth / 1000.0,
                "V2": abs(days_employed) / 1000.0,
                "V3": family_members,
                "V4": emi_paid_off,
                "V5": emi_past_dues,
                "V6": number_of_loans,
                "V7": gender_code,
                "V8": car_code,
                "V9": realty_code,
                "V10": income_code,
                "V11": education_code,
                "V12": family_code,
                "V13": housing_code,
                "V14": annual_income / 100000.0,
                "V15": days_birth / 10000.0,
                "V16": abs(days_employed) / 10000.0,
                "V17": family_members / 10.0,
                "V18": emi_paid_off / 10.0,
                "V19": emi_past_dues / 10.0,
                "V20": number_of_loans / 10.0,
                "V21": annual_income / 1000.0 + emi_paid_off,
                "V22": gender_code + car_code + realty_code,
                "V23": income_code + education_code,
                "V24": family_code + housing_code,
                "V25": family_members + number_of_loans,
                "V26": emi_paid_off + emi_past_dues,
                "V27": annual_income / 10000.0,
                "V28": abs(days_employed) / 10000.0,
                "Amount": annual_income / 1000.0 + emi_paid_off + emi_past_dues,
            }

            feature_vector = [feature_map.get(col, 0.0) for col in FEATURE_ORDER]

            if model is None:
                raise FileNotFoundError("The trained model file is not available.")

            df_features = pd.DataFrame([feature_vector], columns=FEATURE_ORDER)
            pred_raw = model.predict(df_features)
            try:
                proba = model.predict_proba(df_features)[0].tolist()
            except Exception:
                proba = None
            prediction = int(pred_raw[0])
            prediction_message = "Credit Card Approved" if prediction == 1 else "Credit Card Rejected"
            return render_template("result.html", prediction=prediction_message, probability=proba, features=dict(zip(FEATURE_ORDER, feature_vector)))

        except ValueError as exc:
            return render_template("result.html", prediction=str(exc))
        except Exception as exc:
            return render_template("result.html", prediction=f"Prediction error: {exc}")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
