import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Insurance Lapse Risk Tool", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_models():
    with open('logistic_lapse_model.pkl', 'rb') as f:
        log_data = pickle.load(f)
    with open('linear_premium_model.pkl', 'rb') as f:
        lin_data = pickle.load(f)
    return log_data, lin_data

log_data, lin_data = load_models()

st.title("🛡️ Insurance Policy Lapse Risk Tool")
st.markdown("**Powered by AI | For Manager Use Only**")
st.markdown("*Tied to IRDAI 2026 EoM and Commission Reform*")
st.divider()

st.subheader("Enter Customer Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    annual_income = st.number_input("Annual Income (Rs.)", min_value=10000, max_value=1000000, value=60000, step=5000)
    monthly_premium = st.number_input("Monthly Premium (Rs.)", min_value=10, max_value=5000, value=200)
    coverage_amount = st.number_input("Coverage Amount (Rs.)", min_value=10000, max_value=10000000, value=500000, step=10000)
    tenure_months = st.number_input("Policy Tenure (Months)", min_value=1, max_value=360, value=24)

with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])
    marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single", "Widowed"])
    policy_type = st.selectbox("Policy Type", ["Term Life", "Universal Life", "Variable Life", "Whole Life"])
    health_status = st.selectbox("Health Status", ["Excellent", "Good", "Fair"])
    smoking_status = st.selectbox("Smoking Status", ["Non-smoker", "Former smoker", "Current smoker"])
    number_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)

health_map = {"Excellent": 0, "Good": 1, "Fair": 2}
smoke_map = {"Non-smoker": 0, "Former smoker": 1, "Current smoker": 2}
health_score = health_map[health_status]
smoke_score = smoke_map[smoking_status]
annual_premium = monthly_premium * 12
premium_to_income_pct = (annual_premium / annual_income) * 100

if st.button("🔍 Check Lapse Risk", use_container_width=True):
    log_cols = log_data['columns']
    row = {col: 0 for col in log_cols}
    row['age'] = age
    row['number_of_dependents'] = number_of_dependents
    row['annual_income'] = annual_income
    row['health_score'] = health_score
    row['smoke_score'] = smoke_score
    row['coverage_amount'] = coverage_amount
    row['monthly_premium'] = monthly_premium
    row['premium_to_income_pct'] = premium_to_income_pct
    row['tenure_months'] = tenure_months
    if gender == "Male" and 'gender_Male' in row:
        row['gender_Male'] = 1
    if marital_status == "Married" and 'marital_status_Married' in row:
        row['marital_status_Married'] = 1
    elif marital_status == "Single" and 'marital_status_Single' in row:
        row['marital_status_Single'] = 1
    elif marital_status == "Widowed" and 'marital_status_Widowed' in row:
        row['marital_status_Widowed'] = 1
    if policy_type == "Universal Life" and 'policy_type_Universal Life' in row:
        row['policy_type_Universal Life'] = 1
    elif policy_type == "Variable Life" and 'policy_type_Variable Life' in row:
        row['policy_type_Variable Life'] = 1
    elif policy_type == "Whole Life" and 'policy_type_Whole Life' in row:
        row['policy_type_Whole Life'] = 1

    X_input = pd.DataFrame([row])[log_cols]
    X_scaled = log_data['scaler'].transform(X_input)
    prob = log_data['model'].predict_proba(X_scaled)[0][1]
    prob_pct = round(prob * 100, 1)

    if prob_pct >= 60:
        risk_level = "🔴 HIGH RISK"
    elif prob_pct >= 35:
        risk_level = "🟠 MEDIUM RISK"
    else:
        risk_level = "🟢 LOW RISK"

    st.divider()
    st.subheader("Prediction Results")
    st.markdown(f"### {risk_level}")
    st.markdown(f"**Lapse Probability: {prob_pct}%**")

    st.markdown("**Top Reasons for This Risk Score:**")
    reasons = []
    if premium_to_income_pct > 3:
        reasons.append(f"Premium is {round(premium_to_income_pct,1)}% of annual income — affordability stress")
    if policy_type == "Term Life":
        reasons.append("Term Life policies have higher lapse rates than Whole or Universal Life")
    if age < 35:
        reasons.append(f"Younger customer (age {age}) — higher lapse tendency")
    if smoke_score == 2:
        reasons.append("Current smoker status increases lapse risk")
    if tenure_months < 12:
        reasons.append("New policy under 12 months — early-duration lapse risk")
    if marital_status in ["Single", "Divorced"]:
        reasons.append(f"{marital_status} customers show slightly higher lapse tendency")
    if not reasons:
        reasons = ["Premium burden is moderate", "Policy type is stable", "Customer profile is low risk"]
    for i, r in enumerate(reasons[:3], 1):
        st.markdown(f"{i}. {r}")

    lin_cols = lin_data['columns']
    row2 = {col: 0 for col in lin_cols}
    row2['age'] = age
    row2['number_of_dependents'] = number_of_dependents
    row2['annual_income'] = annual_income
    row2['health_score'] = health_score
    row2['smoke_score'] = smoke_score
    row2['coverage_amount'] = coverage_amount
    if gender == "Male" and 'gender_Male' in row2:
        row2['gender_Male'] = 1
    if marital_status == "Married" and 'marital_status_Married' in row2:
        row2['marital_status_Married'] = 1
    elif marital_status == "Single" and 'marital_status_Single' in row2:
        row2['marital_status_Single'] = 1
    elif marital_status == "Widowed" and 'marital_status_Widowed' in row2:
        row2['marital_status_Widowed'] = 1
    if policy_type == "Universal Life" and 'policy_type_Universal Life' in row2:
        row2['policy_type_Universal Life'] = 1
    elif policy_type == "Variable Life" and 'policy_type_Variable Life' in row2:
        row2['policy_type_Variable Life'] = 1
    elif policy_type == "Whole Life" and 'policy_type_Whole Life' in row2:
        row2['policy_type_Whole Life'] = 1

    X_lin_input = pd.DataFrame([row2])[lin_cols]
    est_premium = lin_data['model'].predict(X_lin_input)[0]
    est_premium = max(10, round(est_premium, 2))

    st.divider()
    st.markdown(f"**Estimated Fair Monthly Premium: Rs. {est_premium}**")
    if monthly_premium > est_premium * 1.2:
        st.markdown("⚠️ Current premium is higher than estimated fair value")
    elif monthly_premium < est_premium * 0.8:
        st.markdown("✅ Current premium is lower than estimated fair value — good retention signal")
    else:
        st.markdown("✅ Current premium is close to estimated fair value")

    st.divider()
    st.caption("⚠️ This tool supports manager decisions only. Final action rests with the relationship manager.")
