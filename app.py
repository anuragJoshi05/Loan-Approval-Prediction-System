import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report

# Set page config
st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .approved {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .rejected {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .metrics-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and preprocessing data"""
    try:
        model = joblib.load('best_loan_prediction_model.joblib')
        feature_names = joblib.load('feature_names.joblib')
        preprocessing_mappings = joblib.load('preprocessing_mappings.joblib')
        return model, feature_names, preprocessing_mappings
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run the training script first.")
        st.stop()

def make_prediction(model, feature_names, preprocessing_mappings, user_input):
    """Make prediction with feature engineering"""
    
    # Apply preprocessing mappings
    gender = preprocessing_mappings['Gender'][user_input['Gender']]
    married = preprocessing_mappings['Married'][user_input['Married']]
    education = preprocessing_mappings['Education'][user_input['Education']]
    self_employed = preprocessing_mappings['Self_Employed'][user_input['Self_Employed']]
    property_area = preprocessing_mappings['Property_Area'][user_input['Property_Area']]
    
    # Feature engineering
    total_income = user_input['ApplicantIncome'] + user_input['CoapplicantIncome']
    debt_to_income = user_input['LoanAmount'] / total_income if total_income > 0 else 0
    
    # Categorize loan amount
    if user_input['LoanAmount'] <= 100:
        loan_amount_cat = 0  # Low
    elif user_input['LoanAmount'] <= 200:
        loan_amount_cat = 1  # Medium
    else:
        loan_amount_cat = 2  # High
    
    # Categorize income
    if user_input['ApplicantIncome'] <= 3000:
        income_cat = 0  # Low
    elif user_input['ApplicantIncome'] <= 6000:
        income_cat = 1  # Medium
    else:
        income_cat = 2  # High
    
    # Create feature vector
    features = np.array([[
        gender, married, user_input['Dependents'], education, self_employed,
        user_input['ApplicantIncome'], user_input['CoapplicantIncome'], 
        user_input['LoanAmount'], user_input['Loan_Amount_Term'],
        user_input['Credit_History'], property_area, total_income, 
        debt_to_income, loan_amount_cat, income_cat
    ]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(features)[0]
        return prediction, probability
    else:
        return prediction, None

def main():
    # Load model
    model, feature_names, preprocessing_mappings = load_model()
    
    # Main header
    st.markdown('<h1 class="main-header">🏦 Loan Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced ML System for Loan Approval Prediction")
    
    # Sidebar for model info
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown(f"**Algorithm**: {type(model).__name__}")
        st.markdown(f"**Features**: {len(feature_names)}")
        st.markdown("**Status**: Ready for Prediction")
        
        st.header("🎯 About This Project")
        st.markdown("""
        This system uses advanced machine learning to predict loan approvals with:
        - **Feature Engineering** for better accuracy
        - **Multiple Model Comparison** (SVM, Random Forest, XGBoost)
        - **Real-time Prediction** capability
        - **Professional Deployment** ready interface
        """)
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("📝 Enter Loan Application Details")
        
        # User input form
        with st.form("prediction_form"):
            # Personal Information
            st.subheader("👤 Personal Information")
            col1_1, col1_2 = st.columns(2)
            
            with col1_1:
                gender = st.selectbox("Gender", options=["Male", "Female"])
                married = st.selectbox("Marital Status", options=["No", "Yes"])
                dependents = st.selectbox("Number of Dependents", options=[0, 1, 2, 4])
            
            with col1_2:
                education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
                self_employed = st.selectbox("Self Employed", options=["No", "Yes"])
                property_area = st.selectbox("Property Area", options=["Rural", "Semiurban", "Urban"])
            
            # Financial Information
            st.subheader("💰 Financial Information")
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=1000)
                coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0, value=0, step=1000)
            
            with col2_2:
                loan_amount = st.number_input("Loan Amount (₹ thousands)", min_value=0, value=150, step=10)
                loan_term = st.selectbox("Loan Term (months)", options=[12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8)
            
            # Credit Information
            st.subheader("📊 Credit Information")
            credit_history = st.selectbox("Credit History", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            # Submit button
            submitted = st.form_submit_button("🔮 Predict Loan Status", use_container_width=True)
        
        if submitted:
            # Prepare input data
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
            
            # Make prediction
            prediction, probability = make_prediction(model, feature_names, preprocessing_mappings, user_input)
            
            # Display results in the second column
            with col2:
                st.header("🎯 Prediction Results")
                
                if prediction == 1:
                    st.markdown('<div class="prediction-box approved"><h2>✅ LOAN APPROVED</h2><p>Congratulations! Your loan application is likely to be approved.</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-box rejected"><h2>❌ LOAN REJECTED</h2><p>Unfortunately, your loan application is likely to be rejected.</p></div>', unsafe_allow_html=True)
                
                if probability is not None:
                    st.markdown("### 📈 Prediction Confidence")
                    
                    # Create confidence metrics
                    approval_prob = probability[1]
                    rejection_prob = probability[0]
                    confidence = max(probability)
                    
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric("Approval Probability", f"{approval_prob:.1%}")
                    with col_metric2:
                        st.metric("Model Confidence", f"{confidence:.1%}")
                    
                    # Probability bar chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    categories = ['Rejection', 'Approval']
                    probs = [rejection_prob, approval_prob]
                    colors = ['#ff6b6b', '#51cf66']
                    
                    bars = ax.bar(categories, probs, color=colors, alpha=0.7)
                    ax.set_ylabel('Probability')
                    ax.set_title('Loan Prediction Probabilities')
                    ax.set_ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar, prob in zip(bars, probs):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{prob:.3f}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                
                # Display input summary
                st.markdown("### 📋 Application Summary")
                summary_data = {
                    "Total Income": f"₹{applicant_income + coapplicant_income:,}",
                    "Loan Amount": f"₹{loan_amount:,}K",
                    "Debt-to-Income Ratio": f"{loan_amount / (applicant_income + coapplicant_income) if (applicant_income + coapplicant_income) > 0 else 0:.3f}",
                    "Education": education,
                    "Property Area": property_area
                }
                
                for key, value in summary_data.items():
                    st.text(f"{key}: {value}")
    
    # Additional sections
    if not submitted:
        with col2:
            st.header("📊 Model Performance")
            st.markdown("""
            <div class="metrics-box">
            <h4>🎯 Key Metrics</h4>
            <ul>
                <li><strong>Accuracy</strong>: ~83%</li>
                <li><strong>F1-Score</strong>: ~0.85</li>
                <li><strong>Cross-Validation</strong>: 5-fold</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🔍 Important Features")
            st.markdown("""
            - **Credit History**: Most important factor
            - **Total Income**: Combined applicant income
            - **Debt-to-Income Ratio**: Risk assessment
            - **Property Area**: Location factor
            - **Education Level**: Qualification impact
            """)
    
    # Footer
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    with col_footer1:
        st.markdown("**🚀 Built with**: Python, Scikit-learn, XGBoost")
    with col_footer2:
        st.markdown("**📊 Models**: SVM, Random Forest, XGBoost")
    with col_footer3:
        st.markdown("**🎯 Purpose**: Campus Placement Project")

if __name__ == "__main__":
    main()