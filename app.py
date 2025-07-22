import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dark theme configuration
st.set_page_config(
    page_title="Loan Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a minimal Google-like dark theme
st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: #121212;
            color: #e0e0e0;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-header {
            font-size: 2.5rem;
            color: #90caf9;
            text-align: center;
            margin-bottom: 2rem;
        }
        .prediction-box {
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .approved {
            background-color: #1b5e20;
            color: #ffffff;
        }
        .rejected {
            background-color: #b71c1c;
            color: #ffffff;
        }
        .metrics-box {
            background-color: #1e1e1e;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #90caf9;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_loan_prediction_model.joblib')
        feature_names = joblib.load('feature_names.joblib')
        preprocessing_mappings = joblib.load('preprocessing_mappings.joblib')
        return model, feature_names, preprocessing_mappings
    except FileNotFoundError:
        st.error("Model files not found. Ensure model and preprocessing files exist.")
        st.stop()

def make_prediction(model, feature_names, preprocessing_mappings, user_input):
    gender = preprocessing_mappings['Gender'][user_input['Gender']]
    married = preprocessing_mappings['Married'][user_input['Married']]
    education = preprocessing_mappings['Education'][user_input['Education']]
    self_employed = preprocessing_mappings['Self_Employed'][user_input['Self_Employed']]
    property_area = preprocessing_mappings['Property_Area'][user_input['Property_Area']]

    total_income = user_input['ApplicantIncome'] + user_input['CoapplicantIncome']
    debt_to_income = user_input['LoanAmount'] / total_income if total_income > 0 else 0

    loan_amount_cat = 0 if user_input['LoanAmount'] <= 100 else 1 if user_input['LoanAmount'] <= 200 else 2
    income_cat = 0 if user_input['ApplicantIncome'] <= 3000 else 1 if user_input['ApplicantIncome'] <= 6000 else 2

    features = np.array([[gender, married, user_input['Dependents'], education, self_employed,
                          user_input['ApplicantIncome'], user_input['CoapplicantIncome'],
                          user_input['LoanAmount'], user_input['Loan_Amount_Term'],
                          user_input['Credit_History'], property_area, total_income,
                          debt_to_income, loan_amount_cat, income_cat]])

    prediction = model.predict(features)[0]

    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(features)[0]
        return prediction, probability
    return prediction, None

def main():
    model, feature_names, preprocessing_mappings = load_model()

    st.markdown('<h1 class="main-header">Loan Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("### Machine Learning-Based Loan Approval Classifier")

    with st.sidebar:
        st.header("Model Information")
        st.markdown(f"**Algorithm:** {type(model).__name__}")
        st.markdown(f"**Features Used:** {len(feature_names)}")
        st.markdown("**System Status:** Operational")
        st.header("Project Summary")
        st.markdown("""
        - Feature Engineered ML Pipeline  
        - Models Compared: SVM, Random Forest, XGBoost  
        - Real-time Prediction Interface  
        """)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("Enter Application Details")
        with st.form("prediction_form"):
            st.subheader("Personal Details")
            col1a, col1b = st.columns(2)
            with col1a:
                gender = st.selectbox("Gender", ["Male", "Female"])
                married = st.selectbox("Marital Status", ["No", "Yes"])
                dependents = st.selectbox("Number of Dependents", [0, 1, 2, 4])
            with col1b:
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])
                self_employed = st.selectbox("Self Employed", ["No", "Yes"])
                property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

            st.subheader("Financial Details")
            col2a, col2b = st.columns(2)
            with col2a:
                applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=1000)
                coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0, value=0, step=1000)
            with col2b:
                loan_amount = st.number_input("Loan Amount (₹ thousands)", min_value=0, value=150, step=10)
                loan_term = st.selectbox("Loan Term (months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8)

            st.subheader("Credit Information")
            credit_history = st.selectbox("Credit History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

            submitted = st.form_submit_button("Predict Loan Status")

        if submitted:
            user_input = {
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_term,
                'Credit_History': credit_history,
                'Property_Area': property_area
            }

            prediction, probability = make_prediction(model, feature_names, preprocessing_mappings, user_input)

            with col2:
                st.header("Prediction Results")
                if prediction == 1:
                    st.markdown('<div class="prediction-box approved"><h3>Loan Approved</h3></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-box rejected"><h3>Loan Rejected</h3></div>', unsafe_allow_html=True)

                if probability is not None:
                    approval_prob = probability[1]
                    rejection_prob = probability[0]
                    confidence = max(probability)

                    st.subheader("Prediction Probabilities")
                    col_conf1, col_conf2 = st.columns(2)
                    col_conf1.metric("Approval Probability", f"{approval_prob:.1%}")
                    col_conf2.metric("Model Confidence", f"{confidence:.1%}")

                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.bar(['Rejected', 'Approved'], [rejection_prob, approval_prob], color=['#e53935', '#43a047'])
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Probability")
                    st.pyplot(fig)

                st.subheader("Application Summary")
                summary_data = {
                    "Total Income": f"₹{applicant_income + coapplicant_income:,}",
                    "Loan Amount": f"₹{loan_amount:,}K",
                    "Debt-to-Income Ratio": f"{loan_amount / (applicant_income + coapplicant_income) if (applicant_income + coapplicant_income) > 0 else 0:.3f}",
                    "Education": education,
                    "Property Area": property_area
                }
                for k, v in summary_data.items():
                    st.text(f"{k}: {v}")

    if not submitted:
        with col2:
            st.header("Model Performance")
            st.markdown("""
            <div class="metrics-box">
            <strong>Accuracy:</strong> ~83%  
            <strong>F1-Score:</strong> ~0.85  
            <strong>Cross-Validation:</strong> 5-fold  
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Important Features")
            st.markdown("""
            - Credit History  
            - Total Income  
            - Debt-to-Income Ratio  
            - Property Area  
            - Education  
            """)

    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    col_f1.markdown("Developed using: Python, Scikit-learn, XGBoost")
    col_f2.markdown("Models Used: SVM, Random Forest, XGBoost")
    col_f3.markdown("Made by - Anurag Joshi")

if __name__ == "__main__":
    main()
