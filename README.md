# EMIPredict AI

🚀 **Live App:** https://priyadh1-emipredict-ai.streamlit.app
# 💰 EMIPredict AI

An end-to-end machine learning platform that predicts EMI (loan installment) eligibility and safe maximum monthly EMI amount for applicants across 5 lending scenarios.

🚀 **Live App:** [priyadh1-emipredict-ai.streamlit.app](https://priyadh1-emipredict-ai.streamlit.app)

## Overview

- **Dataset:** 404,800 applicant records, 27 features (demographics, employment, housing, expenses, credit history)
- **Tasks:** 3-class classification (EMI eligibility) + regression (max safe monthly EMI)
- **Models:** 6 trained and compared — Logistic Regression, Random Forest, XGBoost (classification) and Linear Regression, Random Forest, XGBoost (regression)
- **Best Results:** 94% classification accuracy · ₹1,094 regression RMSE (R² 97%)

## Tech Stack

Python · Pandas · Scikit-learn · XGBoost · MLflow · Streamlit

## Pipeline

1. Data cleaning & leakage-free preprocessing (train/val/test split before fitting any transforms)
2. Exploratory Data Analysis (15 visualizations + hypothesis testing)
3. Feature engineering (debt-to-income, expense-to-income, disposable income ratios)
4. Model training & comparison (6 models)
5. Experiment tracking & model registry with MLflow
6. Multi-page Streamlit app (Prediction, EDA Dashboard, Model Performance, Admin/Data)
7. Deployment on Streamlit Community Cloud

## App Features

- 🔮 Real-time EMI eligibility & amount prediction
- 📊 Interactive EDA dashboard
- 🏆 Model performance comparison
- 🗂️ Admin data management panel

## Author

Priyadharshini Murugan
