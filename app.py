import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide"
)

# ============================================================
# LOAD MODELS & ARTIFACTS (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_artifacts():
    clf_model = joblib.load("best_classification_model.pkl")
    reg_model = joblib.load("best_regression_model.pkl")
    clf_feature_columns = joblib.load("classification_feature_columns.pkl")
    reg_feature_columns = joblib.load("regression_feature_columns.pkl")
    label_map = joblib.load("classification_label_map.pkl")
    scaler = joblib.load("scaler.pkl")
    scale_cols = joblib.load("scale_cols.pkl")
    education_order = joblib.load("education_order.pkl")
    return clf_model, reg_model, clf_feature_columns, reg_feature_columns, label_map, scaler, scale_cols, education_order

try:
    (clf_model, reg_model, clf_feature_columns, reg_feature_columns,
     label_map, scaler, scale_cols, education_order) = load_artifacts()
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    LOAD_ERROR = str(e)

# ============================================================
# HELPER: turn raw user inputs into model-ready feature row
# ============================================================
def build_feature_row(inputs, target_columns):
    row = {}

    # --- feature engineering (same formulas as notebook Section 6.3) ---
    total_expenses = (inputs["school_fees"] + inputs["college_fees"] + inputs["travel_expenses"]
                       + inputs["groceries_utilities"] + inputs["other_monthly_expenses"])
    debt_to_income = inputs["current_emi_amount"] / (inputs["monthly_salary"] + 1)
    expense_to_income = total_expenses / (inputs["monthly_salary"] + 1)
    disposable_income = inputs["monthly_salary"] - total_expenses - inputs["monthly_rent"]
    loan_to_income = inputs["requested_amount"] / (inputs["monthly_salary"] + 1)

    row.update({
        "age": inputs["age"],
        "education": education_order[inputs["education"]],
        "monthly_salary": inputs["monthly_salary"],
        "years_of_employment": inputs["years_of_employment"],
        "monthly_rent": inputs["monthly_rent"],
        "family_size": inputs["family_size"],
        "dependents": inputs["dependents"],
        "school_fees": inputs["school_fees"],
        "college_fees": inputs["college_fees"],
        "travel_expenses": inputs["travel_expenses"],
        "groceries_utilities": inputs["groceries_utilities"],
        "other_monthly_expenses": inputs["other_monthly_expenses"],
        "current_emi_amount": inputs["current_emi_amount"],
        "credit_score": inputs["credit_score"],
        "bank_balance": inputs["bank_balance"],
        "emergency_fund": inputs["emergency_fund"],
        "requested_amount": inputs["requested_amount"],
        "requested_tenure": inputs["requested_tenure"],
        "debt_to_income": debt_to_income,
        "total_expenses": total_expenses,
        "expense_to_income": expense_to_income,
        "disposable_income": disposable_income,
        "loan_to_income": loan_to_income,
    })

    # --- one-hot encode categorical fields ---
    for col, val in [
        ("gender", inputs["gender"]),
        ("marital_status", inputs["marital_status"]),
        ("employment_type", inputs["employment_type"]),
        ("company_type", inputs["company_type"]),
        ("house_type", inputs["house_type"]),
        ("emi_scenario", inputs["emi_scenario"]),
        ("existing_loans", inputs["existing_loans"]),
    ]:
        dummy_col = f"{col}_{val}"
        row[dummy_col] = 1

    df_row = pd.DataFrame([row])

    # align to the exact columns the model expects, fill missing dummy cols with 0
    df_row = df_row.reindex(columns=target_columns, fill_value=0)

    # scale the numeric columns using the saved scaler
    cols_present = [c for c in scale_cols if c in df_row.columns]
    df_row[cols_present] = scaler.transform(df_row[cols_present])

    return df_row


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("💰 EMIPredict AI")
page = st.sidebar.radio(
    "Navigate",
    ["🔮 Prediction", "📊 EDA Dashboard", "🏆 Model Performance", "🗂️ Admin / Data"]
)

if not MODELS_LOADED:
    st.error(f"Could not load model files. Make sure all .pkl files are in the same folder as app.py.\n\nError: {LOAD_ERROR}")
    st.stop()


# ============================================================
# PAGE 1 — PREDICTION
# ============================================================
if page == "🔮 Prediction":
    st.title("🔮 EMI Eligibility & Amount Prediction")
    st.write("Enter applicant details below to predict EMI eligibility and the maximum safe monthly EMI amount.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal")
        age = st.number_input("Age", 18, 70, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
        family_size = st.number_input("Family Size", 1, 10, 3)
        dependents = st.number_input("Dependents", 0, 8, 1)

    with col2:
        st.subheader("Employment & Housing")
        monthly_salary = st.number_input("Monthly Salary (₹)", 5000, 300000, 50000, step=1000)
        employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
        years_of_employment = st.number_input("Years of Employment", 0.0, 40.0, 5.0)
        company_type = st.selectbox("Company Type", ["Mid-size", "MNC", "Startup", "Large Indian", "Small"])
        house_type = st.selectbox("House Type", ["Rented", "Family", "Own"])
        monthly_rent = st.number_input("Monthly Rent (₹)", 0, 100000, 10000, step=500)

    with col3:
        st.subheader("Expenses")
        school_fees = st.number_input("School Fees (₹)", 0, 50000, 0, step=500)
        college_fees = st.number_input("College Fees (₹)", 0, 50000, 0, step=500)
        travel_expenses = st.number_input("Travel Expenses (₹)", 0, 30000, 3000, step=500)
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0, 50000, 10000, step=500)
        other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", 0, 50000, 5000, step=500)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.subheader("Credit & Savings")
        existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
        current_emi_amount = st.number_input("Current EMI Amount (₹)", 0, 100000, 0, step=500)
        credit_score = st.number_input("Credit Score", 300, 900, 650)

    with col5:
        bank_balance = st.number_input("Bank Balance (₹)", 0, 2000000, 100000, step=1000)
        emergency_fund = st.number_input("Emergency Fund (₹)", 0, 500000, 20000, step=1000)

    with col6:
        st.subheader("Loan Request")
        emi_scenario = st.selectbox("EMI Scenario", ["Personal Loan EMI", "E-commerce Shopping EMI",
                                                        "Education EMI", "Vehicle EMI", "Home Appliances EMI"])
        requested_amount = st.number_input("Requested Amount (₹)", 5000, 2000000, 100000, step=1000)
        requested_tenure = st.number_input("Requested Tenure (months)", 1, 84, 24)

    st.markdown("---")

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        inputs = dict(
            age=age, gender=gender, marital_status=marital_status, education=education,
            monthly_salary=monthly_salary, employment_type=employment_type,
            years_of_employment=years_of_employment, company_type=company_type,
            house_type=house_type, monthly_rent=monthly_rent, family_size=family_size,
            dependents=dependents, school_fees=school_fees, college_fees=college_fees,
            travel_expenses=travel_expenses, groceries_utilities=groceries_utilities,
            other_monthly_expenses=other_monthly_expenses, existing_loans=existing_loans,
            current_emi_amount=current_emi_amount, credit_score=credit_score,
            bank_balance=bank_balance, emergency_fund=emergency_fund,
            emi_scenario=emi_scenario, requested_amount=requested_amount,
            requested_tenure=requested_tenure
        )

        clf_row = build_feature_row(inputs, clf_feature_columns)
        reg_row = build_feature_row(inputs, reg_feature_columns)

        clf_pred = clf_model.predict(clf_row)[0]
        clf_label = label_map[clf_pred]
        reg_pred = reg_model.predict(reg_row)[0]

        st.markdown("## Results")
        r1, r2 = st.columns(2)

        with r1:
            color = {"Eligible": "green", "High_Risk": "orange", "Not_Eligible": "red"}[clf_label]
            st.markdown(f"### Eligibility: :{color}[{clf_label.replace('_', ' ')}]")
            if hasattr(clf_model, "predict_proba"):
                proba = clf_model.predict_proba(clf_row)[0]
                proba_df = pd.DataFrame({
                    "Class": [label_map[i] for i in range(len(proba))],
                    "Probability": proba
                })
                st.bar_chart(proba_df.set_index("Class"))

        with r2:
            st.markdown(f"### Max Safe Monthly EMI: ₹{reg_pred:,.0f}")
            st.caption("This is the model's estimate of a safe, affordable EMI amount based on the applicant's profile.")


# ============================================================
# PAGE 2 — EDA DASHBOARD
# ============================================================
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")

    if os.path.exists("emi_prediction_dataset.csv"):
        df = pd.read_csv("emi_prediction_dataset.csv")
        st.write(f"Dataset: **{df.shape[0]:,} rows** × **{df.shape[1]} columns**")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("EMI Eligibility Distribution")
            fig, ax = plt.subplots()
            df['emi_eligibility'].value_counts().plot(kind='bar', ax=ax, color='#C44E9E')
            st.pyplot(fig)

        with c2:
            st.subheader("EMI Scenario Distribution")
            fig, ax = plt.subplots()
            df['emi_scenario'].value_counts().plot(kind='barh', ax=ax, color='#4C72B0')
            st.pyplot(fig)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Monthly Salary Distribution")
            fig, ax = plt.subplots()
            ax.hist(df['monthly_salary'].dropna(), bins=40, color='orange')
            st.pyplot(fig)

        with c4:
            st.subheader("Credit Score Distribution")
            fig, ax = plt.subplots()
            ax.hist(df['credit_score'].dropna(), bins=40, color='purple')
            st.pyplot(fig)
    else:
        st.warning("Dataset file (emi_prediction_dataset.csv) not found in this deployment.")


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================
elif page == "🏆 Model Performance":
    st.title("🏆 Model Performance Summary")

    st.subheader("Classification (Target: Accuracy > 90%)")
    clf_summary = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost (Best)"],
        "Test Accuracy": ["76%", "90%", "94%"],
    })
    st.table(clf_summary)

    st.subheader("Regression (Target: RMSE < 2000)")
    reg_summary = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "XGBoost (Best)"],
        "Test RMSE (₹)": ["2,709", "993", "1,094"],
        "Test R²": ["83%", "98%", "97%"],
    })
    st.table(reg_summary)

    st.info("Full experiment history (all params & metrics for all 6 models) is tracked in MLflow (`mlruns/` folder in the project repository).")


# ============================================================
# PAGE 4 — ADMIN / DATA MANAGEMENT (simple CRUD on a CSV log)
# ============================================================
elif page == "🗂️ Admin / Data":
    st.title("🗂️ Admin — Prediction Log")
    st.write("Simple CRUD interface for reviewing and managing logged predictions.")

    log_file = "prediction_log.csv"

    if not os.path.exists(log_file):
        pd.DataFrame(columns=["age", "monthly_salary", "credit_score", "predicted_eligibility", "predicted_emi"]).to_csv(log_file, index=False)

    log_df = pd.read_csv(log_file)

    st.subheader("Current Log")
    st.dataframe(log_df, use_container_width=True)

    st.subheader("Add Entry Manually")
    with st.form("add_entry"):
        a = st.number_input("Age", 18, 70, 30)
        s = st.number_input("Monthly Salary", 5000, 300000, 40000)
        c = st.number_input("Credit Score", 300, 900, 650)
        e = st.selectbox("Predicted Eligibility", ["Eligible", "High_Risk", "Not_Eligible"])
        m = st.number_input("Predicted EMI", 500, 30000, 5000)
        submitted = st.form_submit_button("Add")
        if submitted:
            new_row = pd.DataFrame([[a, s, c, e, m]], columns=log_df.columns)
            log_df = pd.concat([log_df, new_row], ignore_index=True)
            log_df.to_csv(log_file, index=False)
            st.success("Entry added.")
            st.rerun()

    st.subheader("Delete Entry")
    if len(log_df) > 0:
        idx_to_delete = st.number_input("Row index to delete", 0, max(len(log_df) - 1, 0), 0)
        if st.button("Delete Row"):
            log_df = log_df.drop(index=idx_to_delete).reset_index(drop=True)
            log_df.to_csv(log_file, index=False)
            st.success("Entry deleted.")
            st.rerun()
