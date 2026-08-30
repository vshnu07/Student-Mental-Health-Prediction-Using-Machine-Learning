# Student Mental Health Prediction Using Social Media Data

A machine learning project that predicts a student's Mental Health Score (0–10) from social media usage patterns and lifestyle habits, deployed as a live web app for real-time predictions.

## Live Demo
- **Frontend:** https://student-mental-health-prediction-using-sif9.onrender.com
- **Backend API:** https://student-mental-health-prediction-using.onrender.com

## Overview
This project explores how factors like daily screen time, sleep, study hours, physical activity, and stress level relate to a student's mental well-being, then trains a regression model to predict a wellness score from those inputs.

- **Dataset:** 5,000 student records across 13 features (demographic, digital-habit, and lifestyle attributes)
- **Target variable:** Mental Health Score (continuous, 0–10 scale)

## Data Cleaning & Preprocessing
- Removed duplicate records
- Clipped invalid negative values (e.g., physical activity hours)
- Treated outliers using the IQR method
- Reduced country cardinality by grouping 190+ countries into the top 10 plus an "Other" category
- Built a `ColumnTransformer` pipeline:
  - Log-transform + scaling for skewed features (Study Hours)
  - Standard scaling for other numeric features
  - Ordinal encoding for Stress Level (Low → Very High)
  - One-hot encoding for nominal features (Gender, Academic Level, Platform, Purpose of Use, Country)

## Modeling
Two models were trained on a 70/30 train-test split and compared:

| Model | Test R² | Train R² | MAE | RMSE |
|---|---|---|---|---|
| Linear Regression | 0.740 | 0.724 | 0.536 | 0.676 |
| Random Forest (default) | **0.878** | 0.981 | **0.347** | **0.464** |
| Random Forest (tuned) | 0.865 | 0.955 | 0.369 | 0.487 |

- Random Forest outperformed Linear Regression by **13.8 percentage points** in R² and cut RMSE by roughly **31%**.
- Hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`) were tuned via `RandomizedSearchCV` with 5-fold cross-validation across 15 parameter combinations.
- Final pipeline serialized with `Joblib` for deployment.

## Deployment
The trained pipeline is served through a backend API, with a form-based frontend that collects a user's profile and lifestyle inputs and returns a predicted mental health score in real time. Both are hosted on Render.

## Tech Stack
Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Pipelines, ColumnTransformer, Joblib, Jupyter Notebook

## Disclaimer
This tool is built for informational and educational purposes only. It is **not** a clinical assessment or diagnostic tool. If you're struggling with your mental health, please reach out to a professional or someone you trust.
