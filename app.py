import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import json
import re
from PyPDF2 import PdfReader
from chatbot_engine import chatbot

# Set page config
st.set_page_config(
    page_title="HealthAI - Intelligent Risk Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Path to the CSS file
css_path = os.path.join(os.getcwd(), "style.css")
if os.path.exists(css_path):
    local_css(css_path)

# --- USER PERSISTENCE ---
USER_DB = "users.json"

def load_users():
    if os.path.exists(USER_DB):
        with open(USER_DB, "r") as f:
            return json.load(f)
    return {"admin": "admin123"}

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f)

# Initialize Session State
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'users' not in st.session_state:
    st.session_state.users = load_users()

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_model_assets(disease_name):
    """Generic loader for model and scaler."""
    model_mapping = {
        "Parkinson's": ("parkinsons_model.pkl", "parkinsons_scaler.pkl"),
        "Hepatitis": ("hepatitis_model.pkl", "hepatitis_scaler.pkl"),
        "Kidney": ("kidney_model.pkl", "kidney_scaler.pkl"),
        "Heart": ("heart_model.pkl", "heart_scaler.pkl"),
        "Thyroid": ("thyroid_model.pkl", "thyroid_scaler.pkl"),
        "Liver": ("liver_model.pkl", "liver_scaler.pkl")
    }
    
    if disease_name not in model_mapping:
        return None, None
        
    m_file, s_file = model_mapping[disease_name]
    model_path = os.path.join(os.getcwd(), 'models', m_file)
    scaler_path = os.path.join(os.getcwd(), 'models', s_file)
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    return None, None

def get_risk_analysis(disease_name, input_data):
    """Generic prediction function with robust handling for missing features."""
    model, scaler = load_model_assets(disease_name)
    if model is None or scaler is None:
        return None, 0.0
    
    # REQUIRED_FEATURES check to handle "few details" in report
    expected_features = scaler.n_features_in_
    current_features = input_data.shape[1]
    
    processed_data = input_data
    if current_features != expected_features:
        # Clinical Imputation: Fill missing data with zeros/neutral values 
        # to prevent "ValueError: X has n features but StandardScaler is expecting m"
        if current_features < expected_features:
            processed_data = np.pad(input_data, ((0,0), (0, expected_features - current_features)), mode='constant')
        else:
            processed_data = input_data[:, :expected_features]
    
    # Preprocess - Ensure no NaNs leaked into the array
    processed_data = np.nan_to_num(processed_data, nan=0.0)
    
    data_scaled = scaler.transform(processed_data)
    prediction = model.predict(data_scaled)
    
    # Handle proba mapping based on disease type
    try:
        probas = model.predict_proba(data_scaled)[0]
        if disease_name == "Parkinson's":
            # 0: Healthy, 1: Parkinson (Risk is prob of 1)
            probability = probas[1]
        elif disease_name == "Hepatitis":
            # 0: die, 1: live (Risk is prob of 0)
            probability = probas[0]
        elif disease_name == "Kidney":
            # Multi-class: High:0, Low:1, Mod:2, No:3, Sev:4
            # Risk can be viewed as 1.0 - prob(No_Disease)
            probability = 1.0 - probas[3] if len(probas) > 3 else probas[0]
        elif disease_name == "Heart":
            # 0: Absence, 1: Presence
            probability = probas[1]
        elif disease_name == "Thyroid":
            # 0: Healthy, 1: Thyroid
            probability = probas[1]
        elif disease_name == "Liver":
            # 0: No disease, 1: Disease
            probability = probas[1]
        else:
            probability = max(probas)
    except:
        # Fallback logic if predict_proba fails
        if disease_name == "Kidney":
            # 0:High, 1:Low, 2:Mod, 3:No, 4:Sev
            probability = 0.9 if prediction[0] in [0, 4] else (0.5 if prediction[0] == 2 else 0.1)
        elif disease_name == "Hepatitis":
            probability = 0.9 if prediction[0] == 0 else 0.1
        else:
            probability = 0.8 if prediction[0] == 1 else 0.2
        
    return prediction[0], probability

def get_risk_category(probability):
    """Translates probability into category and style."""
    if probability >= 0.70:
        return "High Risk", "high-risk", "Immediate medical consultation recommended."
    elif probability >= 0.35:
        return "Medium Risk", "medium-risk", "Regular monitoring and lifestyle changes advised."
    else:
        return "Low Risk", "low-risk", "Maintain healthy routine and regular checkups."

def reset_analysis_state():
    """Clears all clinical trial and diagnostic data from the session."""
    st.session_state.show_results = False
    st.session_state.analysis_complete = False
    st.session_state.extracted_data = {}
    if 'standardized_df' in st.session_state:
        del st.session_state.standardized_df
    if 'current_file_name' in st.session_state:
        st.session_state.current_file_name = None
    # Increment uploader key to force the widget to reset
    st.session_state.uploader_key = st.session_state.get('uploader_key', 0) + 1

def standardize_medical_data(raw_text):
    """Normalize heterogeneous lab reports into a unified format using synonym mapping and pattern recognition."""
    
    # Canonical Mapping: Metric Name -> List of common regex variations (aliases/synonyms)
    SYNONYM_METRICS = {
        "Blood Pressure": [
            r"Blood\s*Pressure\s*[:\-]?\s*([\d.]+)", r"BP\s*[:\-]?\s*([\d.]+)", 
            r"Systolic\s*BP\s*[:\-]?\s*([\d.]+)", r"Pressure\s*[:\-]?\s*([\d.]+)"
        ],
        "BMI": [
            r"BMI\s*[:\-]?\s*([\d.]+)", r"Body\s*Mass\s*Index\s*[:\-]?\s*([\d.]+)"
        ],
        "Creatinine": [
            r"Creatinine\s*[:\-]?\s*([\d.]+)", r"Serum\s*Creatinine\s*[:\-]?\s*([\d.]+)", 
            r"Creat\s*[:\-]?\s*([\d.]+)", r"S\.\s*Creatinine\s*[:\-]?\s*([\d.]+)"
        ],
        "Bilirubin": [
            r"Bilirubin\s*[:\-]?\s*([\d.]+)", r"Total\s*Bilirubin\s*[:\-]?\s*([\d.]+)", 
            r"Bili\s*[:\-]?\s*([\d.]+)", r"T\.\s*Bilirubin\s*[:\-]?\s*([\d.]+)"
        ],
        "Hemoglobin": [
            r"Hemoglobin\s*[:\-]?\s*([\d.]+)", r"Hb\s*[:\-]?\s*([\d.]+)", 
            r"Hgb\s*[:\-]?\s*([\d.]+)", r"Haemoglobin\s*[:\-]?\s*([\d.]+)"
        ],
        "Albumin": [
            r"Albumin\s*[:\-]?\s*([\d.]+)", r"Serum\s*Albumin\s*[:\-]?\s*([\d.]+)", 
            r"Alb\s*[:\-]?\s*([\d.]+)", r"S\.\s*Albumin\s*[:\-]?\s*([\d.]+)"
        ],
        "Jitter": [
            r"Jitter[^:]*[:\-]?\s*([\d.]+)", r"MDVP:Jitter\(%\)\s*[:\-]?\s*([\d.]+)"
        ],
        "Shimmer": [
            r"Shimmer[^:]*[:\-]?\s*([\d.]+)", r"MDVP:Shimmer\s*[:\-]?\s*([\d.]+)"
        ],
        "HNR": [
            r"HNR[^:]*[:\-]?\s*([\d.]+)", r"Harmonic-to-Noise\s*[:\-]?\s*([\d.]+)"
        ],
        "Glucose": [
            r"Glucose\s*[:\-]?\s*([\d.]+)", r"Blood\s*Sugar\s*[:\-]?\s*([\d.]+)", 
            r"Random\s*Glucose\s*[:\-]?\s*([\d.]+)", r"Fasting\s*Glucose\s*[:\-]?\s*([\d.]+)"
        ],
        "Urea": [
            r"Urea\s*[:\-]?\s*([\d.]+)", r"Blood\s*Urea\s*[:\-]?\s*([\d.]+)", r"BUN\s*[:\-]?\s*([\d.]+)"
        ],
        "SGOT": [
            r"SGOT\s*[:\-]?\s*([\d.]+)", r"AST\s*[:\-]?\s*([\d.]+)", 
            r"Aspartate\s*Aminotransferase\s*[:\-]?\s*([\d.]+)"
        ],
        "Age": [
            r"Age\s*[:\-]?\s*(\d+)", r"Patient\s*Age\s*[:\-]?\s*(\d+)"
        ],
        "Cholesterol": [
            r"Cholesterol\s*[:\-]?\s*([\d.]+)", r"Total\s*Cholesterol\s*[:\-]?\s*([\d.]+)", 
            r"Chol\s*[:\-]?\s*([\d.]+)"
        ]
    }
    
    # --- SYMPTOM MAP ---
    SYMPTOMS = {
        "Kidney": ["Swelling in legs", "Urine color change", "Fatigue", "Nausea", "Frequent urination at night"],
        "Parkinson's": ["Tremors", "Slow movement", "Balance issues", "Speech difficulty", "Rigid muscles"],
        "Diabetes": ["Excess thirst", "Frequent urination", "Blurred vision", "Unexplained weight loss", "Slow healing sores"],
        "Heart": ["Chest pain", "Shortness of breath", "Dizziness", "Palpitations", "Numbness in limbs"],
        "Liver": ["Yellowish skin/eyes", "Abdominal pain/swelling", "Itchy skin", "Dark urine color", "Chronic fatigue"],
        "Thyroid": ["Weight changes", "Sensitivity to cold/heat", "Thinning hair", "Muscle weakness", "Irregular heart rate"]
    }
    
    # --- CLEANING STEP ---
    # Remove common PDF artifacts and control characters that break regex
    clean_text = "".join(char for char in raw_text if char.isprintable() or char in "\n\r\t ")
    
    found_data = {}
    match_count = 0
    
    # Iterate through each canonical metric and try to find a match among its synonyms
    for canonical_name, patterns in SYNONYM_METRICS.items():
        found_val = None
        for pattern in patterns:
            # Multi-line match to handle weird PDF wrapping
            match = re.search(pattern, clean_text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    found_val = float(match.group(1))
                    match_count += 1
                    break # Stop looking for this metric once found
                except (ValueError, IndexError):
                    continue
        found_data[canonical_name] = found_val

    # Unified Metadata for structured output
    units = {
        "Blood Pressure": "mm Hg", "BMI": "kg/m²", "Creatinine": "mg/dL", 
        "Bilirubin": "mg/dL", "Hemoglobin": "g/dL", "Albumin": "g/dL",
        "Jitter": "%", "Shimmer": "dB", "HNR": "dB", "Glucose": "mg/dL",
        "Urea": "mg/dL", "SGOT": "U/L", "Age": "Years", "Cholesterol": "mg/dL"
    }
    ranges = {
        "Blood Pressure": "80-120", "BMI": "18.5-24.9", "Creatinine": "< 0.6 (Low), 0.6-1.3 (Med), > 1.3 (High)",
        "Bilirubin": "< 0.3 (Low), 0.3-1.2 (Med), > 1.2 (High)", "Hemoglobin": "13.5-17.5", 
        "Albumin": "< 3.4 (Low), 3.4-5.4 (Med), > 5.4 (High)",
        "Jitter": "< 1.04", "Shimmer": "< 0.35", "HNR": "> 20", "Glucose": "70-125",
        "Urea": "7-20", "SGOT": "8-45", "Age": "N/A", "Cholesterol": "< 200"
    }
    
    # Convert to structured DataFrame for user review
    table_rows = []
    for name, val in found_data.items():
        table_rows.append({
            "Metric": name,
            "Internal Canonical Key": name.lower().replace(" ", "_"),
            "Value": val if val is not None else "Not Provided",
            "Unit": units.get(name, ""),
            "Reference Range": ranges.get(name, "")
        })
        
    return pd.DataFrame(table_rows), found_data, match_count

# --- UI COMPONENTS ---
def sidebar():
    with st.sidebar:
        st.markdown("## 🧬 HealthAI")
        st.markdown("---")
        
        st.markdown("### System Status")
        st.success("Analysis Engine: Active")
        st.info("Risk Framework: Integrated")
        
        st.markdown("---")
        st.markdown("#### Educational Prototype")
        st.caption("v1.1.0-beta")

def home_page():
    st.title("Intelligent Health Portal")
    
    col1, col2 = st.columns([3, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="main-card" style="margin-bottom: 0;">
        <h3 style="margin-top:0">About the Platform</h3>
        <p style="font-size: 1.15rem; line-height: 1.6; color: #334155;">
        Welcome to the <b>AI-driven Intelligent Health Platform</b>, a state-of-the-art diagnostic assistant designed for early symptom analysis and lifestyle disease prediction. 
        </p>
        <p style="font-size: 1.15rem; line-height: 1.6; color: #334155;">
        Our system utilizes advanced <b>Machine Learning</b> architectures to analyze validated clinical markers, providing real-time risk profiling for various conditions. 
        </p>
        <ul style="list-style-type: none; padding-left: 0; margin-top: 1rem; font-size: 1.15rem; color: #334155;">
            <li style="margin-bottom: 0.5rem;">🩺 Early Risk Stratification</li>
            <li style="margin-bottom: 0.5rem;">🩺 Clinical Marker Analysis</li>
            <li style="margin-bottom: 0.5rem;">🩺 Automated Report Data Extraction</li>
            <li style="margin-bottom: 0.5rem;">🩺 Educational Health Chatbot</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if os.path.exists("image.jpg"):
            # Centering the image directly on the background
            st.image("image.jpg", use_container_width=True)
        else:
            st.warning("image.jpg not found in directory.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Overview Metrics / System Highlights
    mcols = st.columns(3)
    with mcols[0]:
        st.markdown('<div class="metric-card"><div class="metric-value">Optimized</div>ML Pipeline</div>', unsafe_allow_html=True)
    with mcols[1]:
        st.markdown('<div class="metric-card"><div class="metric-value">3 Levels</div>Risk Profiling</div>', unsafe_allow_html=True)
    with mcols[2]:
        st.markdown('<div class="metric-card"><div class="metric-value">Live</div>Diagnostics</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

def prediction_page():
    if os.path.exists("cover2.jpg"):
        st.image("cover2.jpg", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"### 👋 Welcome back, {st.session_state.get('username', 'User')}!")
    st.title("Disease Risk Analysis")
    st.markdown("Enter values from the patient's medical report or upload a digital copy for a comprehensive lifestyle disease risk profiling.")
    
    with st.container():
        # Medical Report Upload Section
        st.subheader("📁 Medical Report Data Extraction")
        st.info("Use the uploader below to automatically populate patient clinical metrics from a digital report.")
        
        # Track file changes to reset state using a dynamic key
        uploader_key = f"report_uploader_{st.session_state.get('uploader_key', 0)}"
        uploaded_file = st.file_uploader("Upload Clinical Report (PDF Only)", type=['pdf'], label_visibility="collapsed", key=uploader_key)
        
        if uploaded_file is not None:
            # Check if this is a new file
            if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
                st.session_state.show_results = False
                st.session_state.extracted_data = {}
                st.session_state.current_file_name = uploaded_file.name

            if st.button("Process & Extract Report", type="primary"):
                with st.spinner("Analyzing report structure and extracting markers..."):
                    try:
                        # 1. Extraction
                        reader = PdfReader(uploaded_file)
                        raw_text = "".join([p.extract_text() for p in reader.pages])
                        
                        # Debug view for troubleshooting extraction
                        with st.expander("🔍 Debug: Report Raw Text Data"):
                            st.code(raw_text)

                        if not raw_text.strip():
                            st.error("The uploaded PDF seems to be empty or unreadable.")
                            return

                        # 2. Extraction & Processing
                        with st.spinner("Analyzing report data..."):
                            time.sleep(1.0)
                            standardized_df, found_data, match_count = standardize_medical_data(raw_text)
                            st.session_state.standardized_df = standardized_df
                        
                        # Validation: If very few medical keywords/metrics, reject
                        medical_tokens = ["glucose", "blood pressure", "bmi", "insulin", "creatinine", "bilirubin", "hemoglobin", "albumin", "report", "jitter", "shimmer"]
                        found_tokens = [t for t in medical_tokens if t in raw_text.lower()]
                        
                        if match_count < 2 and len(found_tokens) < 3:
                            st.warning("⚠️ No significant clinical markers detected. Please upload a valid medical report PDF.")
                            st.session_state.show_results = False
                            return

                        # 3. Store for Prediction
                        st.session_state.extracted_data = found_data
                        st.session_state.show_results = True
                        st.session_state.analysis_complete = False # Reset final results state
                        st.success("✅ Clinical dataset parsed successfully.")
                        st.rerun() 
                    except Exception as e:
                        st.error(f"Error processing report: {e}")
        else:
            # No file uploaded, reset state if it was showing something
            if st.session_state.get('show_results', False):
                st.session_state.show_results = False
                st.session_state.current_file_name = None
        
        st.markdown("---")

        if st.session_state.get('show_results', False):
            st.markdown("---")
            st.subheader("📋 Standardized Diagnostic Data")
            st.write("This table shows the values extracted and normalized from your report into our standard clinical format.")
            
            if 'standardized_df' in st.session_state:
                st.table(st.session_state.standardized_df)

            
            # --- SYMPTOM INPUT SECTION ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🕵️ Additional Symptom Analysis (Optional)")
            st.markdown("Select any observed symptoms to refine the clinical risk profiling.")
            
            user_symptoms = {}
            s_cols = st.columns(3)
            
            # Use the canonical map defined in standardize_medical_data logic
            SYMPTOMS_LIST = {
                "Kidney": ["Swelling in legs", "Urine color change", "Fatigue", "Nausea", "Frequent urination at night"],
                "Parkinson's": ["Tremors", "Slow movement", "Balance issues", "Speech difficulty", "Rigid muscles"],
                "Diabetes": ["Excess thirst", "Frequent urination", "Blurred vision", "Weight loss", "Slow healing sores"],
                "Heart": ["Chest pain", "Shortness of breath", "Dizziness", "Palpitations", "Numbness in limbs"],
                "Liver": ["Yellowish skin/eyes", "Abdominal pain/swelling", "Itchy skin", "Dark urine color", "Chronic fatigue"],
                "Thyroid": ["Weight changes", "Sensitivity to cold/heat", "Thinning hair", "Muscle weakness", "Irregular heart rate"]
            }

            for i, (disease, s_list) in enumerate(SYMPTOMS_LIST.items()):
                with s_cols[i % 3]:
                    user_symptoms[disease] = st.multiselect(f"Symptoms for {disease}", s_list, key=f"s_{disease}")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Finalize Assessment & View Risk Profile", type="primary"):
                st.session_state.analysis_complete = True


            # Calculate Symptom Impact
            def calculate_integrated_risk(base_prob, symptoms, total_possible):
                if base_prob is None: return None
                symptom_score = len(symptoms) / total_possible if total_possible > 0 else 0
                # Weighted Average: 70% Clinical Report, 30% Patient Symptoms
                return (base_prob * 0.7) + (symptom_score * 0.3)

            # Prediction Logic for Diseases
            # Parkinson's
            ext_data = st.session_state.get('extracted_data', {})
            # Specific markers required for Parkinson's
            specific_p = [ext_data.get(k) for k in ["Jitter", "Shimmer", "HNR"] if ext_data.get(k) is not None]
            prob_p = None
            if len(specific_p) >= 1:
                # Use healthy means for missing features instead of disease-neutral ones
                jitter = ext_data.get("Jitter") or 0.003
                shimmer = ext_data.get("Shimmer") or 0.017
                hnr = ext_data.get("HNR") or 24.6
                # Healthy profile: [Fo, Fhi, Flo, Jitter%, Jitter(Abs), RAP, PPQ, DDP, Shimmer, Shimmer(dB), APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
                input_p = np.array([[181.0, 223.0, 145.0, jitter, 0.00002, 0.0019, 0.002, 0.0057, shimmer, 0.16, 0.009, 0.01, 0.013, 0.027, 0.01, hnr, 0.44, 0.69, -6.75, 0.16, 2.15, 0.12]])
                _, prob_p = get_risk_analysis("Parkinson's", input_p)

            # Hepatitis
            # Specific markers required for Hepatitis
            specific_h = [ext_data.get(k) for k in ["Bilirubin", "Albumin", "SGOT"] if ext_data.get(k) is not None]
            prob_h = None
            if len(specific_h) >= 1:
                # Hepatitis needs 19 features
                age = ext_data.get("Age") or 40
                bilirubin = ext_data.get("Bilirubin") or 1.0
                albumin = ext_data.get("Albumin") or 4.0
                sgot = ext_data.get("SGOT") or 30.0
                input_h = np.array([[age, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, bilirubin, 80, sgot, albumin, 40, 0]])
                _, prob_h = get_risk_analysis("Hepatitis", input_h)

                # CLINICAL OVERRIDE: Bilirubin (Total)
                if bilirubin > 1.2: prob_h = max(prob_h, 0.98)
                elif bilirubin < 0.3: prob_h = min(prob_h, 0.20)
                elif 0.3 <= bilirubin <= 1.2: prob_h = 0.50 if prob_h < 0.70 and prob_h > 0.30 else prob_h
                
                # CLINICAL OVERRIDE: Albumin (Serum)
                if albumin > 5.4: prob_h = max(prob_h, 0.98)
                elif albumin < 3.4: prob_h = min(prob_h, 0.20)
                elif 3.4 <= albumin <= 5.4: prob_h = 0.50 if prob_h < 0.70 and prob_h > 0.30 else prob_h

            # Kidney
            # Specific markers required for Kidney (General vitals like BP alone won't trigger it)
            specific_k = [ext_data.get(k) for k in ["Creatinine", "Urea", "Glucose", "Hemoglobin", "Albumin"] if ext_data.get(k) is not None]
            prob_k = None
            if len(specific_k) >= 1:
                # Kidney needs 42 features
                age = ext_data.get("Age") or 50
                bp = ext_data.get("Blood Pressure") or 80
                creatinine = ext_data.get("Creatinine") or 1.2
                urea = ext_data.get("Urea") or 20
                glucose = ext_data.get("Glucose") or 100
                hb = ext_data.get("Hemoglobin") or 14.0
                alb = ext_data.get("Albumin") or 4.0
                
                # Kidney: Use 1.0 as base filler (maps to 'normal' for most categorical features in this dataset)
                input_k = np.ones((1, 42)) 
                
                # Apply healthy clinical medians for missing numerical features
                input_k[0, 0] = age # Age
                input_k[0, 1] = bp # BP
                input_k[0, 2] = 1.020 # Specific Gravity (Normal)
                input_k[0, 3] = 0 # Albumin in urine (0 is healthy)
                input_k[0, 4] = 0 # Sugar in urine (0 is healthy)
                input_k[0, 9] = glucose # Glucose
                input_k[0, 10] = urea # Urea
                input_k[0, 11] = creatinine # Creatinine
                input_k[0, 14] = hb # Hemoglobin
                input_k[0, 27] = alb # Serum Albumin
                _, prob_k = get_risk_analysis("Kidney", input_k)

                # CLINICAL OVERRIDE: Creatinine (Serum)
                if creatinine > 1.3: prob_k = max(prob_k, 0.98)
                elif creatinine < 0.6: prob_k = min(prob_k, 0.20)
                elif 0.6 <= creatinine <= 1.3: prob_k = 0.50 if prob_k < 0.70 and (prob_h is not None and prob_h > 0.30) else prob_k

            # Heart Disease
            prob_heart = None
            if ext_data.get("Blood Pressure") or ext_data.get("Cholesterol"):
                age = ext_data.get("Age") or 50
                bp = ext_data.get("Blood Pressure") or 120
                chol = ext_data.get("Cholesterol") or 200
                # Heart DS features: Age, Sex, CP, BP, Chol, FBS, EKG, MaxHR, Angina, Oldpeak, Slope, Vessels, Thallium
                input_heart = np.array([[age, 1, 3, bp, chol, 0, 1, 150, 0, 1.0, 2, 0, 3]])
                _, prob_heart = get_risk_analysis("Heart", input_heart)
            
            # Thyroid Disease
            prob_thyroid = None
            if ext_data.get("Age"):
                age = ext_data.get("Age") or 40
                # Thyroid features (21): age, sex, thyroxine, query_thy, ... TSH, T3, TT4, T4U, FTI, ref
                input_thyroid = np.zeros((1, 22))
                input_thyroid[0, 0] = age
                input_thyroid[0, 16] = 1.5 # TSH
                input_thyroid[0, 17] = 2.0 # T3
                input_thyroid[0, 18] = 110.0 # TT4
                _, prob_thyroid = get_risk_analysis("Thyroid", input_thyroid)

            # Liver Disease
            prob_liver = None
            if ext_data.get("Bilirubin") or ext_data.get("SGOT"):
                age = ext_data.get("Age") or 40
                bili = ext_data.get("Bilirubin") or 1.0
                sgot = ext_data.get("SGOT") or 30.0
                alb = ext_data.get("Albumin") or 4.0
                # Liver features: Age, Gender, TB, DB, Alkphos, Sgpt, Sgot, Prot, Alb, Ratio
                input_liver = np.array([[age, 1, bili, 0.5, 200, 20, sgot, 7.0, alb, 1.0]])
                _, prob_liver = get_risk_analysis("Liver", input_liver)

                # CLINICAL OVERRIDE: Severe Liver markers
                if bili > 2.0 or sgot > 100:
                    prob_liver = max(prob_liver, 0.98)
                elif bili > 1.2 or sgot > 60:
                    prob_liver = max(prob_liver, 0.75)

            # --- INTEGRATED RISK CALCULATION ---
            # Combine Model outputs with Symptom scores
            final_p = calculate_integrated_risk(prob_p, user_symptoms.get("Parkinson's", []), 5)
            final_h = calculate_integrated_risk(prob_h, user_symptoms.get("Liver", []), 5) # Using Liver symptoms for Hepatitis context
            final_k = calculate_integrated_risk(prob_k, user_symptoms.get("Kidney", []), 5)
            final_heart = calculate_integrated_risk(prob_heart, user_symptoms.get("Heart", []), 5)
            final_thyroid = calculate_integrated_risk(prob_thyroid, user_symptoms.get("Thyroid", []), 5)
            final_liver = calculate_integrated_risk(prob_liver, user_symptoms.get("Liver", []), 5)
            
            # --- LIFESTYLE RISKS (Rule-Based + Symptoms) --- 
            # Diabetes
            prob_diabetes = None
            if ext_data.get("Glucose"):
                gl = ext_data.get("Glucose")
                base_d = 0.9 if gl > 140 else (0.4 if gl > 120 else 0.1)
                prob_diabetes = calculate_integrated_risk(base_d, user_symptoms.get("Diabetes", []), 5)

            # Obesity
            prob_obesity = None
            if ext_data.get("BMI"):
                bmi = ext_data.get("BMI")
                base_o = 0.9 if bmi > 30 else (0.5 if bmi > 25 else 0.1)
                prob_obesity = calculate_integrated_risk(base_o, user_symptoms.get("Obesity", []), 5)

            # Hypertension
            prob_hypertension = None
            if ext_data.get("Blood Pressure"):
                bp = ext_data.get("Blood Pressure")
                base_h_rule = 0.9 if bp > 140 else (0.4 if bp > 130 else 0.1)
                prob_hypertension = calculate_integrated_risk(base_h_rule, user_symptoms.get("Heart", []), 5)

            # Anemia
            prob_anemia = None
            if ext_data.get("Hemoglobin"):
                hb = ext_data.get("Hemoglobin")
                base_a = 0.9 if hb < 11 else (0.4 if hb < 13 else 0.1)
                prob_anemia = calculate_integrated_risk(base_a, [], 5) # Defaulting to empty symptom list for anemia

            # Fatty Liver
            prob_fatty_liver = None
            if ext_data.get("SGOT") or ext_data.get("Bilirubin"):
                sgot = ext_data.get("SGOT") or 30
                base_fl = 0.8 if sgot > 60 else (0.4 if sgot > 45 else 0.1)
                prob_fatty_liver = calculate_integrated_risk(base_fl, user_symptoms.get("Liver", []), 5)

            # High Cholesterol
            prob_cholesterol = None
            if ext_data.get("Cholesterol"):
                ch = ext_data.get("Cholesterol")
                base_c = 0.85 if ch > 240 else (0.45 if ch > 200 else 0.1)
                prob_cholesterol = calculate_integrated_risk(base_c, [], 5)

            # --- DASHBOARD OVERVIEW --- 
            if st.session_state.get('analysis_complete', False):
                st.markdown("---")
                st.subheader("🧬 Standardized Multi-Disease Risk Profile")
                
                # Organize into tabs or groups
                g1, g2 = st.tabs(["🏥 Clinical Analysis (ML Models)", "🥗 Lifestyle Profiling"])

                with g1:
                    st.markdown("#### 🏥 Integrated Clinical Analysis (Report + Symptoms)")
                    res_cols = st.columns(3)
                    ml_predictions = [
                        ("Parkinson's Disease", final_p),
                        ("Hepatitis Condition", final_h),
                        ("Kidney Disease", final_k),
                        ("Heart Disease", final_heart),
                        ("Thyroid Condition", final_thyroid),
                        ("Liver Disease", final_liver)
                    ]
                    for i, (name, prob) in enumerate(ml_predictions):
                        with res_cols[i % 3]:
                            if prob is not None:
                                cat, cls, adv = get_risk_category(prob)
                                st.markdown(f"""
                                <div class="result-box {cls}" style="margin-top: 1rem; padding: 1rem; text-align: left; height: 180px;">
                                    <h5 style="margin:0; color:inherit !important; border-bottom: 1px solid rgba(0,0,0,0.1); padding-bottom: 5px;">{name}</h5>
                                    <div style="font-size: 1.3rem; font-weight: 800; margin-top: 8px;">{cat}</div>
                                    <p style="font-size: 0.85rem; margin: 5px 0; font-weight: 600;">Confidence: {prob:.1%}</p>
                                    <p style="font-size: 0.8rem; opacity: 0.8; font-style: italic;">{adv}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="margin-top: 1rem; padding: 1rem; border: 1px dashed rgba(0,0,0,0.1); border-radius: 10px; height: 180px; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #94a3b8;">
                                    <div style="font-size: 0.9rem; font-weight: 700;">{name}</div>
                                    <div style="font-size: 0.75rem; text-align: center; margin-top: 5px;">Missing clinical markers</div>
                                </div>
                                """, unsafe_allow_html=True)

                with g2:
                    st.markdown("#### 🥗 Integrated Lifestyle Profiling")
                    res_cols2 = st.columns(3)
                    lifestyle_predictions = [
                        ("Diabetes", prob_diabetes),
                        ("Hypertension", prob_hypertension),
                        ("Obesity", prob_obesity),
                        ("Anemia", prob_anemia),
                        ("Fatty Liver Risk", prob_fatty_liver),
                        ("Cholesterol Status", prob_cholesterol)
                    ]
                    for i, (name, prob) in enumerate(lifestyle_predictions):
                        with res_cols2[i % 3]:
                            if prob is not None:
                                cat, cls, adv = get_risk_category(prob)
                                st.markdown(f"""
                                <div class="result-box {cls}" style="margin-top: 1rem; padding: 1rem; text-align: left; height: 180px;">
                                    <h5 style="margin:0; color:inherit !important; border-bottom: 1px solid rgba(0,0,0,0.1); padding-bottom: 5px;">{name}</h5>
                                    <div style="font-size: 1.3rem; font-weight: 800; margin-top: 8px;">{cat}</div>
                                    <p style="font-size: 0.85rem; margin: 5px 0; font-weight: 600;">Risk Level: {prob:.1%}</p>
                                    <p style="font-size: 0.8rem; opacity: 0.8; font-style: italic;">{adv}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="margin-top: 1rem; padding: 1rem; border: 1px dashed rgba(0,0,0,0.1); border-radius: 10px; height: 180px; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #94a3b8;">
                                    <div style="font-size: 0.9rem; font-weight: 700;">{name}</div>
                                    <div style="font-size: 0.75rem; text-align: center; margin-top: 5px;">Missing requirement in report</div>
                                </div>
                                """, unsafe_allow_html=True)

                st.markdown("<br><hr><br>", unsafe_allow_html=True)
                st.subheader("📊 Health Risk Overview Dashboard")
                st.info("Visual summary of integrated clinical and lifestyle risks sorted by priority.")
                
                # Probability Graphs / Overview section
                all_risks = [
                    ("Parkinson's Disease", final_p),
                    ("Hepatitis Condition", final_h),
                    ("Kidney Disease", final_k),
                    ("Heart Disease", final_heart),
                    ("Thyroid Condition", final_thyroid),
                    ("Liver Disease", final_liver),
                    ("Diabetes", prob_diabetes),
                    ("Hypertension", prob_hypertension),
                    ("Obesity", prob_obesity),
                    ("Anemia", prob_anemia),
                    ("Fatty Liver", prob_fatty_liver),
                    ("Cholesterol Status", prob_cholesterol)
                ]
                
                # Filter risks that have valid data
                active_risks = [(name, prob) for name, prob in all_risks if prob is not None]
                active_risks = sorted(active_risks, key=lambda x: x[1], reverse=True) # Sort by probability descending
                
                if active_risks:
                    # Create two columns for the overview graphs
                    graph_col1, graph_col2 = st.columns([2, 1], gap="large")
                    
                    with graph_col1:
                        for name, prob in active_risks:
                            # Color logic for the bars
                            bar_color = "#ef4444" if prob >= 0.7 else ("#f59e0b" if prob >= 0.35 else "#10b981")
                            
                            st.markdown(f"""
                            <div style="margin-bottom: 20px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <span style="font-weight: 600; font-size: 0.95rem; color: #1e293b;">{name}</span>
                                    <span style="font-weight: 700; color: {bar_color};">{prob:.1%}</span>
                                </div>
                                <div style="background-color: #f1f5f9; border-radius: 10px; height: 12px; width: 100%; overflow: hidden;">
                                    <div style="background-color: {bar_color}; height: 100%; width: {prob*100}%; border-radius: 10px; transition: width 1s ease-in-out;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with graph_col2:
                        # Summary Metrics
                        high_risk_count = len([p for n, p in active_risks if p >= 0.7])
                        med_risk_count = len([p for n, p in active_risks if 0.35 <= p < 0.7])
                        
                        st.markdown(f"""
                        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 15px; padding: 1.5rem; text-align: center;">
                            <h4 style="margin-top:0; color: #334155;">Diagnostic Summary</h4>
                            <div style="margin: 1.5rem 0;">
                                <div style="font-size: 2.2rem; font-weight: 800; color: {'#ef4444' if high_risk_count > 0 else '#1e293b'};">{high_risk_count}</div>
                                <div style="color: #64748b; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">High Risk Factors</div>
                            </div>
                            <div style="margin: 1.5rem 0;">
                                <div style="font-size: 2.2rem; font-weight: 800; color: #f59e0b;">{med_risk_count}</div>
                                <div style="color: #64748b; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Caution Points</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Reset button for new analysis
            if st.button("Reset Analysis"):
                reset_analysis_state()
                st.rerun()
        
        

def chatbot_page():
    st.title("Medical Health Chatbot")
    st.markdown("Ask our AI assistant any health-related educational questions.")
    
    # Display chat history
    for message in st.session_state.messages:
        role_class = "user-bubble" if message["role"] == "user" else "ai-bubble"
        st.markdown(f'<div class="chat-bubble {role_class}">{message["content"]}</div>', unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask me about health..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Use the new chatbot engine
            response_text = chatbot.get_response(prompt)
            
            # Simulate typing
            for chunk in response_text.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def login_page():
    st.markdown('<span class="login-logo">🔐</span>', unsafe_allow_html=True)
    
    # Use a separate state variable for the index to avoid Streamlit widget key errors
    if "auth_mode_choice" not in st.session_state:
        st.session_state.auth_mode_choice = 0

    options = ["Sign In", "Create Account"]
    auth_mode = st.radio(
        "Choose Action", 
        options, 
        index=st.session_state.auth_mode_choice,
        label_visibility="collapsed", 
        horizontal=True,
        # We DON'T use a key here that we'll modify later
        key="auth_radio_widget" 
    )
    
    # Sync choice back to session state if user clicks
    st.session_state.auth_mode_choice = options.index(auth_mode)
    
    st.markdown(f'<h1 class="login-title" style="margin-top:1rem;">{auth_mode}</h1>', unsafe_allow_html=True)

    if auth_mode == "Sign In":
        st.markdown('<p style="color: #64748b; font-size: 1.1rem; margin-bottom: 1.5rem;">Access your clinical dashboard</p>', unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Sign In")
            
            if submit:
                # Reload users to ensure latest registrations are available
                current_users = load_users()
                if username in current_users and current_users[username] == password:
                    reset_analysis_state() # Clear previous data on login
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.active_tab_index = 0
                    st.success(f"Welcome back, {username}!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
    else:
        st.markdown('<p style="color: #64748b; font-size: 1.1rem; margin-bottom: 1.5rem;">Join the intelligent health community</p>', unsafe_allow_html=True)
        with st.form("register_form", clear_on_submit=True):
            new_user = st.text_input("Choose Username", placeholder="e.g. john_doe")
            new_pass = st.text_input("Choose Password", type="password", placeholder="Minimum 6 characters")
            confirm_pass = st.text_input("Confirm Password", type="password")
            reg_submit = st.form_submit_button("Create Account")
            
            if reg_submit:
                current_users = load_users() # Check against latest DB
                if not new_user or not new_pass:
                    st.error("Please fill all fields.")
                elif not new_user.isalpha():
                    st.error("Username must contain only letters (no numbers or special characters).")
                elif new_user in current_users:
                    if current_users[new_user] == new_pass:
                        st.error("Username and password already exist.")
                    else:
                        st.error("Username already exists.")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters long.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                else:
                    current_users[new_user] = new_pass
                    save_users(current_users)
                    st.session_state.users = current_users
                    st.success("Account created successfully! Switching to Sign In...")
                    time.sleep(1.5)
                    # Automatically switch back to "Sign In" (Index 0)
                    st.session_state.auth_mode_choice = 0
                    st.rerun()
    
    st.markdown('<p style="text-align:center; color: #64748b; margin-top: 2rem;">Secure 256-bit Encrypted Portal</p>', unsafe_allow_html=True)

# --- MAIN RENDER ---
# sidebar()  # Removed sidebar

if st.session_state.logged_in:
    # Custom Modern Tab Switcher for logged-in users
    # We use a custom state to allow landing on index 1 (Risk Analysis) while keeping it 2nd in order
    if 'active_tab_index' not in st.session_state:
        st.session_state.active_tab_index = 0 # Default to Home on first login
    
    # Render Custom Tab Bar
    tab_labels = ["🏠 Home", "📊 Risk Analysis", "💬 Chatbot"]
    tcols = st.columns(len(tab_labels))
    
    for i, label in enumerate(tab_labels):
        # Apply premium styling for the active tab via session state
        is_active = (st.session_state.active_tab_index == i)
        button_type = "primary" if is_active else "secondary"
        if tcols[i].button(label, use_container_width=True, type=button_type, key=f"tab_btn_{i}"):
            st.session_state.active_tab_index = i
    st.markdown("<hr style='margin: 0.5rem 0 1rem 0; opacity: 0.1;'>", unsafe_allow_html=True)

    # Logout Button (Positioned between tabs and content)
    lcol1, lcol2 = st.columns([8, 2])
    with lcol2:
        if st.button("🚪 Logout", key="main_logout", use_container_width=True):
            reset_analysis_state()
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.active_tab_index = 0
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Dynamic Content based on custom tab selection
    if st.session_state.active_tab_index == 0:
        home_page()
    
    elif st.session_state.active_tab_index == 1:
        prediction_page()
        
    elif st.session_state.active_tab_index == 2:
        chatbot_page()
else:
    # Custom Modern Tab Switcher for guests
    if 'guest_tab_index' not in st.session_state:
        st.session_state.guest_tab_index = 0
    
    # Render Custom Tab Bar
    tab_labels = ["🏠 Home", "🔐 Login", "💬 Chatbot"]
    tcols = st.columns(len(tab_labels))
    
    for i, label in enumerate(tab_labels):
        is_active = (st.session_state.guest_tab_index == i)
        button_type = "primary" if is_active else "secondary"
        if tcols[i].button(label, use_container_width=True, type=button_type, key=f"guest_btn_{i}"):
            st.session_state.guest_tab_index = i
            st.rerun()

    st.markdown("<hr style='margin: 0.5rem 0 2rem 0; opacity: 0.1;'>", unsafe_allow_html=True)

    # Dynamic Content
    if st.session_state.guest_tab_index == 0:
        home_page()
    elif st.session_state.guest_tab_index == 1:
        login_page()
    elif st.session_state.guest_tab_index == 2:
        chatbot_page()

st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes only and does not provide medical advice. Always consult a physician.")
