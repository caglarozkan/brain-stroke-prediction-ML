Brain Stroke Risk Prediction System

This project is a machine learning based web application that predicts
brain stroke risk using patient medical information. The system includes
data analysis, model comparison, imbalanced data handling, threshold
optimization, and API deployment.

The main objective is to build a recall-focused classification model
suitable for medical risk prediction problems, where minimizing false
negatives is critical.


Dataset:

The project uses the “Full Filled Brain Stroke Dataset” from Kaggle.

The dataset is highly imbalanced, with stroke cases forming a small
portion of the data. Because of this, recall was selected as the primary
evaluation metric instead of accuracy.


Methodology:

Data Preprocessing - Numerical features scaled using StandardScaler (for
linear models) - Categorical features encoded using OneHotEncoder -
ColumnTransformer used to ensure consistent preprocessing

Models Evaluated - Logistic Regression - Support Vector Machine (linear
kernel) - Random Forest

Class weighting was applied to address imbalance.

Models were evaluated using: - Recall score (primary metric) - Confusion
matrix - Classification report


Final Model

Logistic Regression was selected due to: - Strong recall performance -
Stable generalization (similar train/test recall) - Interpretability

The classification threshold was adjusted from 0.5 to 0.4 to improve
recall and reduce false negatives.


Deployment

The trained model, preprocessing transformer, and threshold value were
saved using pickle and deployed with FastAPI.

Endpoint: POST /predict

Example response: { “stroke_probability”: 0.63, “threshold_used”: 0.4,
“prediction”: 1, “prediction_label”: “Risk Var” }

A simple HTML/CSS frontend allows users to input patient data and
receive real-time predictions.

Technologies - Python - Pandas, NumPy - Scikit-learn - FastAPI - HTML,
CSS, JavaScript

Running Locally

pip install -r requirements.txt uvicorn app.main:app –reload

Open: http://127.0.0.1:8000
