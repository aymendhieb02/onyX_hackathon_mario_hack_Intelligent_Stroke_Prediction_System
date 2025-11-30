"""
Stroke Risk Prediction - Streamlit App
A modern, user-friendly interface for stroke risk assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Import model wrapper classes (required for loading pickled models)
from model_wrappers import StrokeBinaryPredictor, StrokeProbabilityPredictor

# Page configuration
st.set_page_config(
    page_title="StrokeCare AI - Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .risk-moderate {
        background-color: #fef3c7;
        border-left: 5px solid #eab308;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #d1fae5;
        border-left: 5px solid #22c55e;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODELS
# =============================================================================

@st.cache_resource
def load_models():
    """Load the ML models"""
    binary_model = None
    probability_model = None
    
    try:
        if os.path.exists('stroke_binary_model.pkl'):
            with open('stroke_binary_model.pkl', 'rb') as f:
                binary_model = pickle.load(f)
        else:
            print("⚠️ Binary model file not found")
    except Exception as e:
        print(f"Error loading binary model: {e}")
        binary_model = None
    
    try:
        if os.path.exists('stroke_probability_model.pkl'):
            with open('stroke_probability_model.pkl', 'rb') as f:
                probability_model = pickle.load(f)
        else:
            print("⚠️ Probability model file not found")
    except Exception as e:
        print(f"Error loading probability model: {e}")
        probability_model = None
    
    return binary_model, probability_model

# Load models with error handling (only show spinner on first run)
# Initialize models as None first
binary_model = None
probability_model = None

if 'models_loaded' not in st.session_state:
    try:
        # Try to load models, but don't block if they fail
        binary_model, probability_model = load_models()
        st.session_state['models_loaded'] = True
        st.session_state['binary_model'] = binary_model
        st.session_state['probability_model'] = probability_model
    except Exception as e:
        # Silently fail and use rule-based prediction
        st.session_state['models_loaded'] = True
        st.session_state['binary_model'] = None
        st.session_state['probability_model'] = None
        binary_model = None
        probability_model = None
else:
    # Use cached models from session state
    binary_model = st.session_state.get('binary_model')
    probability_model = st.session_state.get('probability_model')

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_input_dataframe(patient_data):
    """Prepare patient data as DataFrame for the exported models"""
    return pd.DataFrame([{
        'age': float(patient_data.get('age', 0)),
        'gender': patient_data.get('gender', 'Male'),
        'hypertension': int(patient_data.get('hypertension', 0)),
        'heart_disease': int(patient_data.get('heart_disease', 0)),
        'ever_married': patient_data.get('ever_married', 'No'),
        'work_type': patient_data.get('work_type', 'Private'),
        'Residence_type': patient_data.get('residence_type', 'Urban'),
        'avg_glucose_level': float(patient_data.get('avg_glucose_level', 100)),
        'bmi': float(patient_data.get('bmi', 25)),
        'smoking_status': patient_data.get('smoking_status', 'never smoked')
    }])

def calculate_risk_score(patient_data):
    """Calculate rule-based risk score"""
    risk_factors = []
    base_score = 0
    
    age = patient_data['age']
    if age >= 70:
        base_score += 35
        risk_factors.append("Age 70+")
    elif age >= 60:
        base_score += 25
        risk_factors.append("Age 60-69")
    elif age >= 50:
        base_score += 15
        risk_factors.append("Age 50-59")
    
    if patient_data['hypertension']:
        base_score += 20
        risk_factors.append("Hypertension")
    
    if patient_data['heart_disease']:
        base_score += 20
        risk_factors.append("Heart Disease")
    
    glucose = patient_data['avg_glucose_level']
    if glucose >= 200:
        base_score += 15
        risk_factors.append("Very High Glucose")
    elif glucose >= 140:
        base_score += 10
        risk_factors.append("Elevated Glucose")
    
    bmi = patient_data['bmi']
    if bmi >= 35:
        base_score += 10
        risk_factors.append("Severe Obesity")
    elif bmi >= 30:
        base_score += 7
        risk_factors.append("Obesity")
    
    smoking = patient_data['smoking_status']
    if smoking == 'smokes':
        base_score += 15
        risk_factors.append("Current Smoker")
    elif smoking == 'formerly smoked':
        base_score += 5
        risk_factors.append("Former Smoker")
    
    risk_percentage = min(95, max(5, base_score))
    
    if risk_percentage >= 60:
        risk_level = "HIGH"
    elif risk_percentage >= 30:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"
    
    return risk_percentage, risk_level, risk_factors

def predict_stroke_risk(patient_data):
    """Predict stroke risk using ML models"""
    # Get rule-based risk factors (always available)
    rule_percentage, rule_level, risk_factors = calculate_risk_score(patient_data)
    
    # If models not loaded, use rule-based
    if binary_model is None or probability_model is None:
        return rule_percentage, rule_level, risk_factors, None
    
    try:
        input_df = prepare_input_dataframe(patient_data)
        
        # Get binary prediction
        binary_prediction = None
        try:
            binary_prediction = int(binary_model.predict(input_df)[0])
        except Exception as e:
            print(f"Binary prediction error: {e}")
        
        # Get probability prediction
        risk_probability = None
        try:
            risk_probability = probability_model.predict(input_df)[0]
        except Exception as e:
            print(f"Probability prediction error: {e}")
        
        # Use ML prediction if available, otherwise fallback to rule-based
        if risk_probability is not None:
            risk_percentage = int(risk_probability * 100)
            risk_percentage = max(0, min(100, risk_percentage))
            
            # If binary prediction is 1, ensure minimum risk
            if binary_prediction == 1:
                risk_percentage = max(risk_percentage, 30)
            
            # Determine risk level
            if risk_percentage >= 60:
                risk_level = "HIGH"
            elif risk_percentage >= 30:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"
        else:
            # Fallback to rule-based
            risk_percentage = rule_percentage
            risk_level = rule_level
        
        return risk_percentage, risk_level, risk_factors, binary_prediction
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Always return rule-based as fallback
        return rule_percentage, rule_level, risk_factors, None

# =============================================================================
# MAIN APP
# =============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>❤️ StrokeCare AI</h1>
    <p>Intelligent Stroke Risk Assessment • Compassionate Care</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for input
with st.sidebar:
    st.header("📋 Patient Information")
    
    # Personal Information
    st.subheader("👤 Personal")
    age = st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    ever_married = st.selectbox("Marital Status", ["Yes", "No"])
    work_type = st.selectbox("Work Type", [
        "Private", "Self-employed", "Govt_job", "children", "Never_worked"
    ])
    
    # Medical History
    st.subheader("🏥 Medical")
    hypertension = st.checkbox("Hypertension (High Blood Pressure)")
    heart_disease = st.checkbox("Heart Disease")
    avg_glucose_level = st.number_input(
        "Average Glucose Level (mg/dL)", 
        min_value=50.0, 
        max_value=500.0, 
        value=100.0, 
        step=0.1
    )
    bmi = st.number_input(
        "BMI (Body Mass Index)", 
        min_value=10.0, 
        max_value=60.0, 
        value=25.0, 
        step=0.1
    )
    
    # BMI Calculator
    with st.expander("📊 Calculate BMI"):
        height_cm = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
        weight_kg = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
        if st.button("Calculate BMI"):
            if height_cm > 0:
                bmi_calculated = weight_kg / ((height_cm / 100) ** 2)
                st.success(f"Your BMI: {bmi_calculated:.1f}")
                st.info("You can copy this value to the BMI field above.")
    
    # Lifestyle
    st.subheader("🌱 Lifestyle")
    smoking_status = st.selectbox("Smoking Status", [
        "never smoked", "formerly smoked", "smokes", "Unknown"
    ])
    
    # Predict button
    predict_button = st.button("🔍 Assess Stroke Risk", type="primary", use_container_width=True)

# Main content area
if predict_button:
    # Collect patient data
    patient_data = {
        'age': age,
        'gender': gender,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    
    # Show loading with error handling
    try:
        with st.spinner("🔍 Analyzing your health data..."):
            risk_percentage, risk_level, risk_factors, binary_prediction = predict_stroke_risk(patient_data)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please try again or contact support if the issue persists.")
        st.stop()
    
    # Display results
    st.markdown("---")
    st.header("📊 Assessment Results")
    
    # Risk percentage and level
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Risk Percentage", f"{risk_percentage}%")
    
    with col2:
        st.metric("Risk Level", risk_level)
    
    with col3:
        if binary_prediction is not None:
            binary_text = "⚠️ Risk Detected" if binary_prediction == 1 else "✓ No Risk"
            st.metric("ML Prediction", binary_text)
        else:
            st.metric("Model Status", "Rule-Based")
    
    # Risk level indicator
    if risk_level == "HIGH":
        st.markdown(f"""
        <div class="risk-high">
            <h3>⚠️ HIGH RISK</h3>
            <p>Your assessment indicates a high risk level. Please consult with a healthcare professional 
            for a comprehensive evaluation and personalized care plan.</p>
        </div>
        """, unsafe_allow_html=True)
    elif risk_level == "MODERATE":
        st.markdown(f"""
        <div class="risk-moderate">
            <h3>💛 MODERATE RISK</h3>
            <p>Your assessment shows a moderate risk level. While this isn't cause for alarm, 
            it's an opportunity to make positive changes for your health. Consider lifestyle modifications 
            and regular health check-ups.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-low">
            <h3>✓ LOW RISK</h3>
            <p>Great news! Your assessment shows a low risk level. Continue maintaining healthy habits 
            and regular health check-ups.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk factors
    if risk_factors:
        st.subheader("🔍 Key Risk Factors Identified")
        for factor in risk_factors:
            st.markdown(f"- ⚠️ {factor}")
    else:
        st.success("✓ No significant risk factors identified")
    
    # Detailed metrics
    st.subheader("📈 Detailed Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Age</h4>
            <p style="font-size: 1.5rem; font-weight: bold;">{}</p>
        </div>
        """.format(age), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>Glucose Level</h4>
            <p style="font-size: 1.5rem; font-weight: bold;">{} mg/dL</p>
        </div>
        """.format(avg_glucose_level), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>BMI</h4>
            <p style="font-size: 1.5rem; font-weight: bold;">{}</p>
        </div>
        """.format(bmi), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>Medical Conditions</h4>
            <p>Hypertension: {}<br>Heart Disease: {}</p>
        </div>
        """.format("Yes" if hypertension else "No", "Yes" if heart_disease else "No"), 
        unsafe_allow_html=True)
    
    # Recommendations
    st.subheader("💡 Recommendations")
    recommendations = []
    
    if age >= 60:
        recommendations.append("Regular health check-ups are especially important at your age.")
    if hypertension:
        recommendations.append("Monitor your blood pressure daily and keep a log to share with your doctor.")
    if heart_disease:
        recommendations.append("Stay consistent with any prescribed heart medications.")
    if avg_glucose_level > 140:
        recommendations.append("Your glucose levels suggest monitoring is needed. Consider consulting an endocrinologist.")
    if bmi > 30:
        recommendations.append("Gradual weight management through balanced nutrition can significantly reduce your risk.")
    if smoking_status in ['smokes', 'formerly smoked']:
        recommendations.append("If you smoke, quitting is the single most impactful change you can make.")
    
    if not recommendations:
        recommendations.append("Continue maintaining a healthy lifestyle with regular exercise and balanced nutrition.")
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    ⚠️ **Important Disclaimer:** This assessment is for informational purposes only and does not constitute 
    medical advice. Please consult a healthcare professional for proper diagnosis and treatment.
    """)
    
    # Download report button
    st.download_button(
        label="📄 Download Report",
        data=f"""
Stroke Risk Assessment Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Patient Information:
- Age: {age}
- Gender: {gender}
- Residence: {residence_type}
- Marital Status: {ever_married}
- Work Type: {work_type}

Medical History:
- Hypertension: {'Yes' if hypertension else 'No'}
- Heart Disease: {'Yes' if heart_disease else 'No'}
- Average Glucose Level: {avg_glucose_level} mg/dL
- BMI: {bmi}

Lifestyle:
- Smoking Status: {smoking_status}

Assessment Results:
- Risk Percentage: {risk_percentage}%
- Risk Level: {risk_level}
- Binary Prediction: {'Risk Detected' if binary_prediction == 1 else 'No Risk' if binary_prediction == 0 else 'N/A'}

Risk Factors:
{chr(10).join('- ' + factor for factor in risk_factors) if risk_factors else 'None identified'}

Recommendations:
{chr(10).join(str(i+1) + '. ' + rec for i, rec in enumerate(recommendations))}

---
This report is for informational purposes only. Please consult a healthcare professional.
        """,
        file_name=f"stroke_risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

else:
    # Welcome message
    st.info("""
    👋 **Welcome to StrokeCare AI!**
    
    Please fill in the patient information in the sidebar and click "Assess Stroke Risk" to get started.
    
    This tool uses advanced machine learning models to assess stroke risk based on:
    - Personal information (age, gender, etc.)
    - Medical history (hypertension, heart disease, glucose levels, BMI)
    - Lifestyle factors (smoking status)
    
    Your data is processed securely and is not stored.
    """)
    
    # Information cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🔒 Private & Secure</h4>
            <p>Your data is processed securely and not stored.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 AI-Powered</h4>
            <p>Advanced machine learning models for accurate predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>❤️ Human-Centered</h4>
            <p>Compassionate care with actionable recommendations.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem 0;">
    <p>Built with ❤️ for better health outcomes</p>
    <p style="font-size: 0.8rem;">StrokeCare AI © 2024 • Powered by Advanced Machine Learning</p>
</div>
""", unsafe_allow_html=True)

