import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the model and encoders (adjust file names as needed)
model = joblib.load('loan_model.joblib')
# Suppose you have label encoders or mappings as dictionaries
with open('feature_mappings.joblib', 'rb') as f:
    feature_mappings = joblib.load(f)

st.set_page_config(page_title="Loan Eligibility Assessment", layout="centered")

# CSS for a clean, minimalist "Google-like" look
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Roboto', sans-serif;
            background-color: #fafbfc;
        }
        .main .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
        }
        .stButton>button {
            width: 100%;
            font-weight: 500;
            background-color: #1a73e8;
            color: white;
            border-radius: 5px;
            padding: 10px 0;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: #fff;
            border-radius: 4px;
            border: 1px solid #dadce0;
        }
        .stSelectbox>div>div>div {
            border-radius: 4px;
            border: 1px solid #dadce0;
        }
        .prediction-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: #fff;
            box-shadow: 0 2px 12px rgba(60,64,67,0.1);
            padding: 2rem;
            text-align: center;
            margin-top: 2rem;
        }
        .result-approved {
            color: #1a73e8;
            font-weight: 600;
            font-size: 2rem;
        }
        .result-rejected {
            color: #d50000;
            font-weight: 600;
            font-size: 2rem;
        }
        .confidence-score {
            font-size: 1.2rem;
            color: #555;
            margin-top: .5rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Loan Eligibility Assessment")

st.write("Please provide accurate financial and personal details to assess loan eligibility.")

with st.form("loan_form"):
    # Section 1: Applicant Details
    st.subheader("Applicant Details")
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    # Section 2: Financial Details
    st.subheader("Financial Details")
    applicant_income = st.number_input("Applicant Income (Monthly)", min_value=0, max_value=1000000, step=1000)
    coapplicant_income = st.number_input("Co-applicant Income (Monthly)", min_value=0, max_value=1000000, step=1000)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, max_value=1000, step=1)
    loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=12, max_value=480, step=12)
    credit_history = st.selectbox("Credit History", ["1.0 (Good)", "0.0 (Bad)"])
    
    submitted = st.form_submit_button("Submit")

if submitted:
    # Prepare input vector in the order used by the model
    # Encoding categorical features
    # Adjust keys if your model expects different column names
    # Example: gender: {'Male':1, 'Female':0}
    input_dict = {
        "Gender": feature_mappings['Gender'][gender],
        "Married": feature_mappings['Married'][marital_status],
        "Dependents": feature_mappings['Dependents'][dependents],
        "Education": feature_mappings['Education'][education],
        "Self_Employed": feature_mappings['Self_Employed'][self_employed],
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": 1.0 if credit_history.startswith("1") else 0.0,
        "Property_Area": feature_mappings['Property_Area'][property_area]
    }
    features_order = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Property_Area"
    ]
    input_values = np.array([input_dict[key] for key in features_order]).reshape(1, -1)
    prediction = model.predict(input_values)[0]
    confidence = model.predict_proba(input_values)[0][int(prediction)] * 100

    result_text = "Approved" if prediction == 1 else "Rejected"
    result_class = "result-approved" if prediction == 1 else "result-rejected"
    st.markdown(f"""
        <div class="prediction-card">
            <div class="{result_class}">{result_text}</div>
            <div class="confidence-score">Confidence: {confidence:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)
