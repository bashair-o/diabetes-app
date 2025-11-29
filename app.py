import streamlit as st
import pickle
import numpy as np
import time

# Cache model loading for performance
@st.cache_data
def load_models():
    rf_model = pickle.load(open("model_random_forest.pkl", "rb"))
    svm_model = pickle.load(open("model_svm.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return rf_model, svm_model, scaler

rf_model, svm_model, scaler = load_models()

# Helper functions
def yn_to_num(x): return 1 if x == "Yes" else 0
def gender_to_num(g): return 1 if g == "Male" else 0

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical-themed design
st.markdown("""
<style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f8f9fa;
        color: #333;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 20px;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #007bff;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }
    .section {
        background-color: #f1f3f4;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 5px solid #007bff;
    }
    .section-title {
        font-size: 20px;
        font-weight: bold;
        color: #007bff;
        margin-bottom: 10px;
    }
    .input-label {
        font-weight: bold;
        color: #000;
        text-align: center;
        margin-bottom: 5px;
    }
    .stSelectbox, .stNumberInput {
        margin-bottom: 10px;
    }
    .predict-btn {
        background-color: #28a745;
        color: white;
        border: none;
        padding: 12px;
        border-radius: 5px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        cursor: pointer;
    }
    .reset-btn {
        background-color: #6c757d;
        color: white;
        border: none;
        padding: 12px;
        border-radius: 5px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        cursor: pointer;
    }
    .result-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        margin-top: 20px;
    }
    .result-box-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        margin-top: 20px;
    }
    .progress-bar {
        width: 100%;
        height: 20px;
        background-color: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 10px;
    }
    .progress-fill {
        height: 100%;
        background-color: #28a745;
        transition: width 0.5s;
    }
    .progress-fill-danger {
        background-color: #dc3545;
    }
    .footer {
        text-align: center;
        font-size: 12px;
        color: #666;
        margin-top: 40px;
        padding: 15px;
        background-color: #e9ecef;
        border-radius: 5px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    

    
    
    
    
</style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<h1 class='title'>🩺 Diabetes Risk Assessment Tool</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A clinical decision support system for diabetes risk evaluation based on patient symptoms.</p>", unsafe_allow_html=True)

# Sidebar for model selection
with st.sidebar:
    st.header("Model Selection")
    model_choice = st.selectbox("Select Algorithm:", ["Random Forest", "SVM"], help="Choose the machine learning model for prediction.")
    st.markdown("---")
    st.markdown("**Note:** This tool is for informational purposes only. Consult a healthcare professional for diagnosis.")

# Patient Information Section
with st.expander("Patient Demographics", expanded=True):
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>👤 Patient Information</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='input-label'>Age</div>", unsafe_allow_html=True)
        Age = st.number_input("", min_value=1, max_value=120, value=40, help="Enter patient's age in years.")
    with col2:
        st.markdown("<div class='input-label'>Gender</div>", unsafe_allow_html=True)
        Gender = st.selectbox("", ["Male", "Female"], help="Select patient's gender.")
    st.markdown("</div>", unsafe_allow_html=True)

# Symptoms Section (All Reported Symptoms with unique keys)
with st.expander("Symptom Assessment", expanded=True):
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧬 Reported Symptoms</div>", unsafe_allow_html=True)
    colA, colB, colC, colD = st.columns(4)
    
    with colA:
        st.markdown("<div class='input-label'>Polyuria</div>", unsafe_allow_html=True)
        Polyuria = st.selectbox("", ["No", "Yes"], key="polyuria")
        st.markdown("<div class='input-label'>Polydipsia</div>", unsafe_allow_html=True)
        Polydipsia = st.selectbox("", ["No", "Yes"], key="polydipsia")
        st.markdown("<div class='input-label'>Sudden Weight Loss</div>", unsafe_allow_html=True)
        sudden_weight_loss = st.selectbox("", ["No", "Yes"], key="sudden_weight_loss")
        st.markdown("<div class='input-label'>Weakness</div>", unsafe_allow_html=True)
        weakness = st.selectbox("", ["No", "Yes"], key="weakness")
    
    with colB:
        st.markdown("<div class='input-label'>Polyphagia</div>", unsafe_allow_html=True)
        Polyphagia = st.selectbox("", ["No", "Yes"], key="polyphagia")
        st.markdown("<div class='input-label'>Genital Thrush</div>", unsafe_allow_html=True)
        Genital_thrush = st.selectbox("", ["No", "Yes"], key="genital_thrush")
        st.markdown("<div class='input-label'>Visual Blurring</div>", unsafe_allow_html=True)
        visual_blurring = st.selectbox("", ["No", "Yes"], key="visual_blurring")
        st.markdown("<div class='input-label'>Itching</div>", unsafe_allow_html=True)
        Itching = st.selectbox("", ["No", "Yes"], key="itching")
    
    with colC:
        st.markdown("<div class='input-label'>Irritability</div>", unsafe_allow_html=True)
        Irritability = st.selectbox("", ["No", "Yes"], key="irritability")
        st.markdown("<div class='input-label'>Delayed Healing</div>", unsafe_allow_html=True)
        delayed_healing = st.selectbox("", ["No", "Yes"], key="delayed_healing")
        st.markdown("<div class='input-label'>Partial Paresis</div>", unsafe_allow_html=True)
        partial_paresis = st.selectbox("", ["No", "Yes"], key="partial_paresis")
        st.markdown("<div class='input-label'>Muscle Stiffness</div>", unsafe_allow_html=True)
        muscle_stiffness = st.selectbox("", ["No", "Yes"], key="muscle_stiffness")
    
    with colD:
        st.markdown("<div class='input-label'>Alopecia</div>", unsafe_allow_html=True)
        Alopecia = st.selectbox("", ["No", "Yes"], key="alopecia")
        st.markdown("<div class='input-label'>Obesity</div>", unsafe_allow_html=True)
        Obesity = st.selectbox("", ["No", "Yes"], key="obesity")
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction and Reset Buttons
col_pred, col_reset = st.columns(2)
with col_pred:
    predict_button = st.button("Assess Risk", key="predict", help="Click to run the diabetes risk assessment.")
with col_reset:
    if st.button("Reset Form", key="reset", help="Clear all inputs."):
        st.rerun()

# Prediction Logic and Result Display
if predict_button:
    with st.spinner("Processing assessment..."):
        time.sleep(1)
    
    row = np.array([
        Age, gender_to_num(Gender), yn_to_num(Polyuria), yn_to_num(Polydipsia),
        yn_to_num(sudden_weight_loss), yn_to_num(weakness), yn_to_num(Polyphagia),
        yn_to_num(Genital_thrush), yn_to_num(visual_blurring), yn_to_num(Itching),
        yn_to_num(Irritability), yn_to_num(delayed_healing), yn_to_num(partial_paresis),
        yn_to_num(muscle_stiffness), yn_to_num(Alopecia), yn_to_num(Obesity)
    ]).reshape(1, -1)
    
    row = scaler.transform(row)
    model = rf_model if model_choice == "Random Forest" else svm_model
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]
    
    if pred == 0:
        st.markdown(f"""
        <div class='result-box'>
            <h3>✅ Low Risk Assessment</h3>
            <p>Based on the provided symptoms, the patient shows a low likelihood of diabetes.</p>
            <p>Risk Probability: {prob * 100:.1f}%</p>
            <div class='progress-bar'>
                <div class='progress-fill' style='width: {prob * 100}%'></div>
            </div>
            <p><em>Recommendation: Maintain healthy lifestyle habits.</em></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='result-box-danger'>
            <h3>⚠️ High Risk Assessment</h3>
            <p>Based on the provided symptoms, the patient shows an elevated risk of diabetes.</p>
            <p>Risk Probability: {prob * 100:.1f}%</p>
            <div class='progress-bar'>
                <div class='progress-fill progress-fill-danger' style='width: {prob * 100}%'></div>
            </div>
            <p><em>Recommendation: Seek consultation with a healthcare provider for further evaluation.</em></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    <p><strong>Medical Disclaimer:</strong> This tool is not a substitute for professional medical advice, diagnosis, or treatment. Results are based on machine learning models and should be interpreted by qualified healthcare professionals. Always consult a physician for personalized health decisions.</p>
    <p>Developed for educational and research purposes. Data privacy is maintained; no information is stored.</p>
</div>
""", unsafe_allow_html=True)
