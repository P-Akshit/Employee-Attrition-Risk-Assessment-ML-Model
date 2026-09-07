Employee Attrition Risk Assessment API
An end-to-end Machine Learning web service built with Scikit-Learn and FastAPI that predicts employee attrition risk. Instead of returning a plain binary outcome, the API provides probability-based risk labeling (High Risk, Low Risk, or Uncertain) to enable actionable HR decision-making.

Overview
Machine Learning Pipeline: Trained using a RandomForestClassifier optimized via GridSearchCV hyperparameter tuning.

Categorical Handling: Uses LabelEncoder mappings saved as persistent binary artifacts to handle input feature transformations during API requests.

REST API: Powered by FastAPI and Pydantic schema validation, offering lightweight, high-performance inferences with built-in parameter fallback handling.

Dataset & Model Features
The model expects the following 8 input features:

Education (Categorical): Bachelors, Masters, PHD

JoiningYear (Numerical): Year of joining (e.g., 2021)

City (Categorical, Optional): Bangalore, Pune, New Delhi (Default: Bangalore)

PaymentTier (Numerical, Optional): Payment tier status 1, 2, or 3 (Default: 2)

Age (Numerical): Employee age in years

Gender (Categorical, Optional): Male or Female (Default: Male)

EverBenched (Categorical, Optional): Yes or No (Default: No)

ExperienceInCurrentDomain (Numerical): Years of domain experience

Project Structure
Plaintext
.
├── train.py              # ML Training & Hyperparameter Tuning Pipeline
├── main.py               # FastAPI Web Application & Prediction Endpoint
├── model.pkl             # Trained Random Forest Estimator
├── label_encoders.pkl    # Serialized LabelEncoders Dictionary
├── requirements.txt      # Python Dependencies
└── README.md             # Project Documentation
Setup & Installation
1. Prerequisites
Ensure Python 3.8+ is installed on your machine.

2. Clone Repository & Install Dependencies
Bash
git clone https://github.com/your-username/employee-attrition-api.git
cd employee-attrition-api
pip install -r requirements.txt

3. Requirements (requirements.txt)
Plaintext
fastapi
uvicorn
pydantic
scikit-learn
pandas
numpy
joblib
Training the Model
To train the Random Forest Classifier, run hyperparameter grid search, and generate model.pkl and label_encoders.pkl:

Bash
python train.py
Running the API Server
Start the FastAPI application with Uvicorn:

Bash
uvicorn main:app --reload
The server will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

API Usage
Interactive API Documentation
Open your browser and visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test endpoints directly via Swagger UI.

Endpoint: POST /predict
Request Body (JSON)
JSON
{
  "Education": "Bachelors",
  "JoiningYear": 2021,
  "City": "Pune",
  "PaymentTier": 2,
  "Age": 28,
  "Gender": "Male",
  "EverBenched": "No",
  "ExperienceInCurrentDomain": 3
}
Response (JSON)
JSON
{
  "leave_probability": 0.7425,
  "risk_level": "High Risk of Leaving",
  "prediction_code": 1
}
Risk Labeling Logic
The prediction probability (P) for an employee leaving is evaluated using the following thresholds:

High Risk of Leaving: P≥0.51

Low Risk of Leaving: P≤0.49

Uncertain / 50-50 Chance: 0.49<P<0.51
