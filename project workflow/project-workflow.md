# Project Workflow

## Overview
This project delivers a credit approval prediction system using machine learning, applicant data capture, credit history analysis, and a web API for decision delivery. The workflow covers preparation, modelling, prediction, deployment, and operations.

## System Architecture
- User interface or API for application submission.
- Data layer containing applicant and credit history entities.
- Machine learning layer that preprocesses data, trains, validates, and scores applications.
- Persistence layer for saving users, applications, model artifacts, and predictions.
- Deployment layer that exposes prediction services to end users.

## Key Entities and Relationships
- `Users`
  - Stores login information and role (applicant, admin).
  - Related to `Applicant_Details` through `UserID`.
- `Applicant_Details`
  - Contains demographic, employment, education, and income fields.
  - Example fields: `Gender`, `Age`, `IncomeType`, `EducationType`, `FamilyStatus`, `HousingType`, `EmploymentDays`.
  - Each applicant may have one or multiple credit history records.
- `Credit_History`
  - Contains past loan and repayment behavior.
  - Example fields: `CreditType`, `Status`, `MonthlyPayment`, `RemainingDebt`, `Term`.
  - Used to enrich prediction inputs and improve model accuracy.
- `ML_Model`
  - Represents the trained model artifact and metadata.
  - Stores model version, training date, metrics, and feature definitions.
- `Approval_Prediction`
  - Stores prediction results for each application.
  - Includes predicted approval status, probability score, and timestamp.

## Detailed Workflow Steps

### 1. Environment Setup
1. Install required software:
   - Python 3.10 or later
   - Git
   - Optional: Anaconda Navigator, VS Code, PyCharm
2. Create and activate an isolated Python environment.
3. Install required libraries from `requirements.txt` or manually with:
   - `pip install numpy pandas scikit-learn matplotlib seaborn flask`
4. Verify dependencies by importing the libraries in Python.

### 2. Data Ingestion and Validation
1. Identify input sources: user form submissions, CSV files, or database tables.
2. Load raw applicant and credit history data into Pandas DataFrames.
3. Validate data quality:
   - confirm required columns exist
   - verify numeric ranges and categorical values
   - detect duplicate records and invalid entries
4. Store cleaned data in the project database or intermediate files.

### 3. Data Preprocessing
1. Handle missing values via imputation or removal.
2. Encode categorical features:
   - one-hot encode `IncomeType`, `EducationType`, `FamilyStatus`, `HousingType`
   - map ordinal categories if any
3. Scale numeric features such as `EmploymentDays`, `Age`, `MonthlyPayment`, `RemainingDebt`.
4. Engineer derived features from credit history, such as:
   - total number of past loans
   - average repayment ratio
   - number of delinquent accounts
5. Split the dataset into training, validation, and test partitions.

### 4. Model Training and Evaluation
1. Choose one or more algorithms from scikit-learn:
   - logistic regression, decision trees, random forest, gradient boosting
2. Train models using applicant and credit history features.
3. Evaluate performance with metrics such as:
   - accuracy
   - precision, recall, F1-score
   - ROC AUC
4. Perform hyperparameter tuning to improve model generalization.
5. Save the selected model artifact and store training metadata.
6. Document the final model version and evaluation results.

### 5. Prediction and Decision Flow
1. A user submits an application through the frontend or API.
2. Persist the application in `Applicant_Details`.
3. Load related `Credit_History` records for the applicant.
4. Apply the same preprocessing pipeline used during training.
5. Call the trained `ML_Model` to generate a prediction.
6. Save the result in `Approval_Prediction` with:
   - approval status (`approved` / `declined`)
   - confidence score
   - applied model version
7. Return the predicted outcome to the user.

### 6. Deployment and Monitoring
1. Build a Flask web service for prediction requests.
2. Expose endpoints for:
   - application submission
   - prediction retrieval
   - model health checks
3. Deploy the service to a server or cloud host.
4. Monitor model and application performance:
   - API uptime
   - prediction latency
   - accuracy drift over time
5. Plan periodic retraining when new data becomes available.

## Implementation Notes
- Keep data validation and feature engineering consistent between training and prediction.
- Store model metadata to support version tracking and rollbacks.
- Use logging for every request and model decision.
- Implement error handling for invalid inputs and missing credit history.

## Expected Outcomes
- Clean and validated applicant data drives accurate predictions.
- Credit history improves risk assessment.
- Prediction results are stored and auditable.
- The deployed system supports user applications and returns fast loan decisions.

## Future Enhancements
- Add administrator dashboards for model performance and application review.
- Support batch scoring of multiple applicants.
- Introduce explainability output for predictions, such as feature importance.
- Add authentication and role-based access control for secure submissions.
