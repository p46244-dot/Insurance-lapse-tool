"""
Insurance Policy Lapse Prediction Tool
---------------------------------------
Streamlit app for relationship managers to check a customer's lapse risk
(logistic regression) and estimated fair monthly premium (linear regression).

Author: Generated for MBA capstone project
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Policy Lapse Predictor",
    page_icon="📋",
    layout="centered",
)

APP_DIR = Path(__file__).parent
LOGISTIC_MODEL_PATH = APP_DIR / "logistic_lapse_model.pkl"
LINEAR_MODEL_PATH = APP_DIR / "linear_premium_model.pkl"


# --------------------------------------------------------------------------
# LOAD MODELS (cached so they load once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_models():
    with open(LOGISTIC_MODEL_PATH, "rb") as f:
        logistic_bundle = pickle.load(f)
    with open(LINEAR_MODEL_PATH, "rb") as f:
        linear_bundle = pickle.load(f)
    return logistic_bundle, linear_bundle


try:
    logistic_bundle, linear_bundle = load_models()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure 'logistic_lapse_model.pkl' and "
        "'linear_premium_model.pkl' are in the same folder as app.py."
    )
    st.stop()

log_model = logistic_bundle["model"]
log_scaler = logistic_bundle["scaler"]
log_columns = logistic_bundle["columns"]

lin_model = linear_bundle["model"]
lin_columns = linear_bundle["columns"]

# --------------------------------------------------------------------------
# ENCODING HELPERS
# --------------------------------------------------------------------------
# NOTE ON ASSUMPTIONS:
# The two .pkl files only contain the fitted model + scaler + final column
# order — they do not contain the original preprocessing/feature-engineering
# code. The mappings below were reverse-engineered from the scaler's mean_/
# scale_ values and are a reasonable best fit, but please sanity-check them
# against your original training notebook and adjust if needed.

HEALTH_SCORE_MAP = {
    "Excellent": 0,
    "Good": 1,
    "Fair": 2,
    "Poor": 3,
}

SMOKE_SCORE_MAP = {
    "Non-smoker": 0,
    "Smoker": 1,
}

POLICY_TYPES = ["Term Life", "Whole Life", "Universal Life", "Variable Life"]
MARITAL_STATUSES = ["Divorced", "Married", "Single", "Widowed"]
GENDERS = ["Female", "Male", "Other / Prefer not to say"]

RISK_LOW_MAX = 0.33
RISK_MEDIUM_MAX = 0.66


def encode_common(age, dependents, income, health_status, smoker_status,
                   coverage, gender, marital_status, policy_type):
    """Build the shared dict of raw + one-hot engineered features."""
    health_score = HEALTH_SCORE_MAP[health_status]
    smoke_score = SMOKE_SCORE_MAP[smoker_status]

    row = {
        "age": age,
        "number_of_dependents": dependents,
        "annual_income": income,
        "health_score": health_score,
        "smoke_score": smoke_score,
        "coverage_amount": coverage,
        "gender_Male": 1 if gender == "Male" else 0,
        "gender__RARE_": 1 if gender == "Other / Prefer not to say" else 0,
        "marital_status_Married": 1 if marital_status == "Married" else 0,
        "marital_status_Single": 1 if marital_status == "Single" else 0,
        "marital_status_Widowed": 1 if marital_status == "Widowed" else 0,
        "policy_type_Universal Life": 1 if policy_type == "Universal Life" else 0,
        "policy_type_Variable Life": 1 if policy_type == "Variable Life" else 0,
        "policy_type_Whole Life": 1 if policy_type == "Whole Life" else 0,
    }
    return row


def build_logistic_input(row, monthly_premium, tenure_months, income):
    row = dict(row)
    row["monthly_premium"] = monthly_premium
    row["premium_to_income_pct"] = (monthly_premium * 12 / income) * 100 if income > 0 else 0
    row["tenure_months"] = tenure_months
    df = pd.DataFrame([row])[log_columns]
    return df


def build_linear_input(row):
    df = pd.DataFrame([row])[lin_columns]
    return df


def risk_bucket(prob):
    if prob < RISK_LOW_MAX:
        return "Low", "🟢"
    elif prob < RISK_MEDIUM_MAX:
        return "Medium", "🟡"
    else:
        return "High", "🔴"


# Plain-English explanation text per engineered feature.
# Each entry: (label_if_higher_than_average, label_if_lower_than_average)
FEATURE_EXPLANATIONS = {
    "age": ("Customer is older than the typical policyholder", "Customer is younger than the typical policyholder"),
    "number_of_dependents": ("Customer has more dependents than average", "Customer has fewer dependents than average"),
    "annual_income": ("Customer's annual income is above average", "Customer's annual income is below average"),
    "health_score": ("Customer's health status is a concern", "Customer's health status is favorable"),
    "smoke_score": ("Customer is a smoker", "Customer is a non-smoker"),
    "coverage_amount": ("Coverage amount is higher than typical", "Coverage amount is lower than typical"),
    "monthly_premium": ("Monthly premium is higher than typical", "Monthly premium is lower than typical"),
    "premium_to_income_pct": ("Premium takes up a large share of income", "Premium takes up a small share of income"),
    "tenure_months": ("Customer has a longer policy tenure", "Customer is relatively new to the policy"),
    "gender_Male": ("Gender: Male", "Gender: Not Male"),
    "gender__RARE_": ("Gender: Other/Unspecified", "Gender: Male or Female"),
    "marital_status_Married": ("Marital status: Married", "Marital status: Not Married"),
    "marital_status_Single": ("Marital status: Single", "Marital status: Not Single"),
    "marital_status_Widowed": ("Marital status: Widowed", "Marital status: Not Widowed"),
    "policy_type_Universal Life": ("Policy type: Universal Life", "Policy type: Not Universal Life"),
    "policy_type_Variable Life": ("Policy type: Variable Life", "Policy type: Not Variable Life"),
    "policy_type_Whole Life": ("Policy type: Whole Life", "Policy type: Not Whole Life"),
}


def top_reasons(input_df, n=3):
    """
    Returns the top n features pushing the prediction toward LAPSE (class 1),
    based on each feature's contribution (coef * scaled value) to the
    logistic regression's log-odds.
    """
    scaled = log_scaler.transform(input_df)
    contributions = scaled[0] * log_model.coef_[0]

    order = np.argsort(contributions)[::-1]  # most positive (pushes toward lapse) first
    reasons = []
    for idx in order:
        if contributions[idx] <= 0:
            break  # only show features actually pushing toward lapse
        feature_name = log_columns[idx]
        is_above_mean = scaled[0][idx] > 0
        explanation_pair = FEATURE_EXPLANATIONS.get(feature_name, (feature_name, feature_name))
        text = explanation_pair[0] if is_above_mean else explanation_pair[1]
        reasons.append(text)
        if len(reasons) == n:
            break

    if not reasons:
        reasons = ["No single factor stands out — overall profile is close to average risk."]

    return reasons


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("📋 Insurance Policy Lapse Predictor")
st.caption("Manager decision-support tool — enter customer details to assess lapse risk.")

with st.form("customer_form"):
    st.subheader("Customer Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        annual_income = st.number_input("Annual Income (₹)", min_value=0, value=600000, step=10000)
        monthly_premium = st.number_input("Monthly Premium (₹)", min_value=0, value=2500, step=100)
        coverage_amount = st.number_input("Coverage Amount (₹)", min_value=0, value=1000000, step=50000)
        tenure_months = st.number_input("Tenure (months)", min_value=0, value=12, step=1)
    with col2:
        policy_type = st.selectbox("Policy Type", POLICY_TYPES)
        marital_status = st.selectbox("Marital Status", MARITAL_STATUSES)
        smoker_status = st.selectbox("Smoking Status", list(SMOKE_SCORE_MAP.keys()))
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=15, value=1, step=1)
        health_status = st.selectbox("Health Status", list(HEALTH_SCORE_MAP.keys()))

    gender = st.selectbox("Gender", GENDERS)
    st.caption(
        "Gender was added because the underlying model requires it as an input, "
        "even though it wasn't in the original requested field list."
    )

    submitted = st.form_submit_button("Check Lapse Risk", type="primary", use_container_width=True)

if submitted:
    common_row = encode_common(
        age, dependents, annual_income, health_status, smoker_status,
        coverage_amount, gender, marital_status, policy_type,
    )

    # ---- Logistic regression: lapse risk ----
    log_input = build_logistic_input(common_row, monthly_premium, tenure_months, annual_income)
    lapse_prob = log_model.predict_proba(log_scaler.transform(log_input))[0][1]
    risk_label, risk_emoji = risk_bucket(lapse_prob)
    reasons = top_reasons(log_input, n=3)

    # ---- Linear regression: fair premium ----
    lin_input = build_linear_input(common_row)
    fair_premium = lin_model.predict(lin_input)[0]

    st.divider()
    st.subheader("Results")

    risk_color = {"Low": "green", "Medium": "orange", "High": "red"}[risk_label]
    st.markdown(
        f"### {risk_emoji} Lapse Risk: :{risk_color}[{risk_label}]  "
        f"—  **{lapse_prob * 100:.1f}%** probability"
    )
    st.progress(min(max(lapse_prob, 0.0), 1.0))

    st.markdown("**Top reasons for this flag:**")
    for i, reason in enumerate(reasons, start=1):
        st.markdown(f"{i}. {reason}")

    st.markdown("---")
    st.markdown(
        f"**Estimated fair monthly premium:** ₹{fair_premium:,.2f}  \n"
        f"*(Customer's actual premium: ₹{monthly_premium:,.2f})*"
    )
    diff = monthly_premium - fair_premium
    if abs(diff) > 0.05 * max(fair_premium, 1):
        direction = "above" if diff > 0 else "below"
        st.caption(f"Current premium is {direction} the model's fair-value estimate by ₹{abs(diff):,.2f}.")

    st.divider()
    st.info(
        "**Disclaimer:** This tool supports manager decisions only — "
        "final action rests with the relationship manager."
    )
