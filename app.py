"""
⚠️ DEPRECATED: This Streamlit app is being phased out in favor of the new
React + FastAPI architecture located at:
  - Backend: api/main.py (run with: uvicorn api.main:app --reload)
  - Frontend: frontend/ (run with: cd frontend && npm run dev)

This file is kept for backward compatibility and reference.
For new development, please use the FastAPI backend.
"""
import warnings
warnings.warn(
    "app.py (Streamlit) is deprecated. Use the new React frontend + FastAPI backend instead.",
    DeprecationWarning
)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_model_and_preprocessor, get_feature_schema, get_categorical_options, make_prediction
import config

# Page Config
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.PAGE_CONFIG['layout'],
    initial_sidebar_state=config.PAGE_CONFIG['initial_sidebar_state']
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .dropout { background-color: #ffcccc; color: #990000; }
    .enrolled { background-color: #ffffcc; color: #999900; }
    .graduate { background-color: #ccffcc; color: #006600; }
</style>
""", unsafe_allow_html=True)

# Header
st.title(f"{config.APP_ICON} {config.APP_TITLE}")
st.markdown("### Predict student outcomes based on academic and demographic data.")
st.divider()

# Load Resources
@st.cache_resource
def load_resources():
    return load_model_and_preprocessor()

@st.cache_data
def load_schema():
    return get_feature_schema()

try:
    model, preprocessor, target_encoder = load_resources()
    num_cols, cat_cols, all_cols = load_schema()
    
    # Sidebar - Inputs
    st.sidebar.header("📝 Student Information")
    
    # Add info expander
    with st.sidebar.expander("ℹ️ How to use this form"):
        st.markdown("""
        **Binary Fields (0 or 1):**
        - 0 = No/Female/Rural
        - 1 = Yes/Male/Urban
        
        **Tips:**
        - Enter `0` for negative/false values
        - Enter `1` for positive/true values
        - Use realistic ranges for grades and rates
        """)
    
    input_data = {}
    
    # Define feature metadata
    feature_info = {
        'Age': {'min': 17, 'max': 70, 'default': 20, 'help': 'Age at enrollment (years)'},
        'Gender': {'min': 0, 'max': 1, 'default': 0, 'help': '0 = Female, 1 = Male'},
        'Marital_Status': {'min': 0, 'max': 6, 'default': 1, 'help': '1 = Single, 2 = Married, etc.'},
        'Course': {'min': 0, 'max': 20, 'default': 1, 'help': 'Course ID (varies by institution)'},
        'Mother_Qualification': {'min': 0, 'max': 45, 'default': 1, 'help': 'Education level code'},
        'Father_Qualification': {'min': 0, 'max': 45, 'default': 1, 'help': 'Education level code'},
        'Previous_Qualification': {'min': 0, 'max': 20, 'default': 1, 'help': 'Prior education code'},
        'Admission_Grade': {'min': 0.0, 'max': 200.0, 'default': 100.0, 'help': 'Admission grade score'},
        'Displaced': {'min': 0, 'max': 1, 'default': 0, 'help': '0 = No, 1 = Yes (living away from home)'},
        'Debtor': {'min': 0, 'max': 1, 'default': 0, 'help': '0 = No unpaid fees, 1 = Has debt'},
        'Tuition_Fees_Up_To_Date': {'min': 0, 'max': 1, 'default': 1, 'help': '0 = Behind, 1 = Paid'},
        'Scholarship_Holder': {'min': 0, 'max': 1, 'default': 0, 'help': '0 = No scholarship, 1 = Has scholarship'},
        'Unemployment_Rate': {'min': 0.0, 'max': 30.0, 'default': 10.0, 'help': 'Unemployment % in region'},
        'Inflation_Rate': {'min': -5.0, 'max': 10.0, 'default': 1.0, 'help': 'Inflation rate %'},
        'GDP': {'min': -10.0, 'max': 10.0, 'default': 0.0, 'help': 'GDP growth rate %'},
    }
    
    with st.sidebar.form("prediction_form"):
        # Demographics
        st.subheader("👤 Demographics")
        if 'Age' in all_cols:
            info = feature_info.get('Age', {})
            input_data['Age'] = st.number_input(
                "Age", 
                min_value=info.get('min', 0), 
                max_value=info.get('max', 100),
                value=info.get('default', 20),
                help=info.get('help', '')
            )
        
        if 'Gender' in all_cols:
            info = feature_info.get('Gender', {})
            input_data['Gender'] = st.selectbox(
                "Gender",
                options=[0, 1],
                format_func=lambda x: "Female" if x == 0 else "Male",
                help=info.get('help', '')
            )
        
        if 'Marital_Status' in all_cols:
            info = feature_info.get('Marital_Status', {})
            input_data['Marital_Status'] = st.number_input(
                "Marital Status Code",
                min_value=info.get('min', 0),
                max_value=info.get('max', 6),
                value=info.get('default', 1),
                help=info.get('help', '')
            )
        
        # Academic
        st.subheader("🎓 Academic Info")
        for col in ['Course', 'Mother_Qualification', 'Father_Qualification', 
                    'Previous_Qualification', 'Admission_Grade']:
            if col in all_cols:
                info = feature_info.get(col, {})
                input_data[col] = st.number_input(
                    col.replace('_', ' '),
                    min_value=float(info.get('min', 0)),
                    max_value=float(info.get('max', 200)),
                    value=float(info.get('default', 0)),
                    help=info.get('help', '')
                )
        
        # Financial
        st.subheader("💰 Financial Status")
        for col in ['Displaced', 'Debtor', 'Tuition_Fees_Up_To_Date', 'Scholarship_Holder']:
            if col in all_cols:
                info = feature_info.get(col, {})
                input_data[col] = st.selectbox(
                    col.replace('_', ' '),
                    options=[0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    help=info.get('help', ''),
                    key=col
                )
        
        # Economic Indicators
        st.subheader("📊 Economic Indicators")
        for col in ['Unemployment_Rate', 'Inflation_Rate', 'GDP']:
            if col in all_cols:
                info = feature_info.get(col, {})
                input_data[col] = st.number_input(
                    col.replace('_', ' '),
                    min_value=float(info.get('min', -10)),
                    max_value=float(info.get('max', 30)),
                    value=float(info.get('default', 0)),
                    help=info.get('help', ''),
                    step=0.1
                )
            
        submit_button = st.form_submit_button("🔮 Predict Outcome")

    # Main Content - Prediction
    if submit_button:
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        
        # Ensure column order matches
        input_df = input_df[all_cols]
        
        # Predict
        with st.spinner("Analyzing student data..."):
            pred_idx, probabilities = make_prediction(model, preprocessor, input_df)
            
        # Decode Prediction
        pred_label = target_encoder.inverse_transform([pred_idx])[0]
        
        # Display Result
        st.subheader("Prediction Result")
        
        if pred_label == 'Dropout':
            st.markdown(f'<div class="prediction-box dropout"><h2>🔴 Dropout</h2><p>High risk of dropping out.</p></div>', unsafe_allow_html=True)
        elif pred_label == 'Enrolled':
            st.markdown(f'<div class="prediction-box enrolled"><h2>🟡 Enrolled</h2><p>Likely to continue enrollment.</p></div>', unsafe_allow_html=True)
        else: # Graduate
            st.markdown(f'<div class="prediction-box graduate"><h2>🟢 Graduate</h2><p>Likely to graduate successfully.</p></div>', unsafe_allow_html=True)
            
        # Probability Visualization
        st.subheader("Confidence Scores")
        prob_df = pd.DataFrame({
            'Outcome': target_encoder.classes_,
            'Probability': probabilities
        })
        
        # Bar Chart
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=prob_df, x='Probability', y='Outcome', palette=['#ff9999', '#ffff99', '#99ff99'], ax=ax)
        ax.set_xlim(0, 1)
        for i, v in enumerate(probabilities):
            ax.text(v + 0.01, i, f"{v:.1%}", va='center')
        st.pyplot(fig)
        
        # Feature Values Summary
        with st.expander("View Input Data Summary"):
            st.dataframe(input_df)

    else:
        st.info("👈 Please fill in the student details in the sidebar and click 'Predict Outcome'.")
        
        # Show model info
        st.markdown("#### Model Information")
        st.write(f"**Model Type:** {type(model).__name__}")
        st.write(f"**Features Used:** {len(all_cols)}")
        st.write(f"**Target Classes:** {', '.join(target_encoder.classes_)}")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.warning("Please ensure the model and data are correctly processed.")

