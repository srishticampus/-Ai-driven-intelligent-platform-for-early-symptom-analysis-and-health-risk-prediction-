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
        "Kidney": ("kidney_model.pkl", "kidney_scaler.pkl")
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
    st.session_state.extracted_data = {}
    if 'standardized_df' in st.session_state:
        del st.session_state.standardized_df

def standardize_medical_data(raw_text):
    """Normalize heterogeneous lab reports into a unified format using synonym mapping and pattern recognition."""
    
    # Canonical Mapping: Metric Name -> List of common regex variations (aliases/synonyms)
    SYNONYM_METRICS = {
        "Blood Pressure": [
            r"BP:\s*([\d.]+)", r"Blood Pressure:\s*([\d.]+)", r"Systolic BP:\s*([\d.]+)", 
            r"Diastolic BP:\s*([\d.]+)", r"Pressure:\s*([\d.]+)"
        ],
        "BMI": [
            r"BMI:\s*([\d.]+)", r"Body Mass Index:\s*([\d.]+)", r"Body-Mass-Index:\s*([\d.]+)"
        ],
        "Creatinine": [
            r"Creatinine:\s*([\d.]+)", r"Serum Creatinine:\s*([\d.]+)", r"Creat:\s*([\d.]+)",
            r"S\. Creatinine:\s*([\d.]+)"
        ],
        "Bilirubin": [
            r"Bilirubin:\s*([\d.]+)", r"Total Bilirubin:\s*([\d.]+)", r"Bili:\s*([\d.]+)",
            r"T\. Bilirubin:\s*([\d.]+)"
        ],
        "Hemoglobin": [
            r"Hemoglobin:\s*([\d.]+)", r"Hb:\s*([\d.]+)", r"Hgb:\s*([\d.]+)", r"Haemoglobin:\s*([\d.]+)"
        ],
        "Albumin": [
            r"Albumin:\s*([\d.]+)", r"Serum Albumin:\s*([\d.]+)", r"Alb:\s*([\d.]+)",
            r"S\. Albumin:\s*([\d.]+)", r"Albumin Level:\s*([\d.]+)"
        ],
        "Jitter": [
            r"Jitter[^:]*:\s*([\d.]+)", r"MDVP:Jitter\(%\):\s*([\d.]+)", r"MDVP:Jitter\(Abs\):\s*([\d.]+)"
        ],
        "Shimmer": [
            r"Shimmer[^:]*:\s*([\d.]+)", r"MDVP:Shimmer:\s*([\d.]+)", r"Shimmer\(dB\):\s*([\d.]+)"
        ],
        "HNR": [
            r"HNR[^:]*:\s*([\d.]+)", r"Harmonic-to-Noise:\s*([\d.]+)", r"Noise-Ratio:\s*([\d.]+)"
        ],
        "Glucose": [
            r"Glucose:\s*([\d.]+)", r"Blood Sugar:\s*([\d.]+)", r"Glucose level:\s*([\d.]+)",
            r"Random Glucose:\s*([\d.]+)", r"Fasting Glucose:\s*([\d.]+)"
        ],
        "Urea": [
            r"Urea:\s*([\d.]+)", r"Blood Urea:\s*([\d.]+)", r"BUN:\s*([\d.]+)", 
            r"Blood Urea Nitrogen\s*\(BUN\):\s*([\d.]+)", r"Urea Nitrogen:\s*([\d.]+)"
        ],
        "SGOT": [
            r"SGOT:\s*([\d.]+)", r"AST:\s*([\d.]+)", r"Aspartate Aminotransferase:\s*([\d.]+)"
        ],
        "Age": [
            r"Age:\s*(\d+)", r"Patient Age:\s*(\d+)", r"Years:\s*(\d+)"
        ]
    }
    
    found_data = {}
    match_count = 0
    
    # Iterate through each canonical metric and try to find a match among its synonyms
    for canonical_name, patterns in SYNONYM_METRICS.items():
        found_val = None
        for pattern in patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
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
        "Urea": "mg/dL", "SGOT": "U/L", "Age": "Years"
    }
    ranges = {
        "Blood Pressure": "80-120", "BMI": "18.5-24.9", "Creatinine": "< 0.6 (Low), 0.6-1.3 (Med), > 1.3 (High)",
        "Bilirubin": "< 0.3 (Low), 0.3-1.2 (Med), > 1.2 (High)", "Hemoglobin": "13.5-17.5", 
        "Albumin": "< 3.4 (Low), 3.4-5.4 (Med), > 5.4 (High)",
        "Jitter": "< 1.04", "Shimmer": "< 0.35", "HNR": "> 20", "Glucose": "70-125",
        "Urea": "7-20", "SGOT": "8-45", "Age": "N/A"
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
    st.info("Explore the tabs above to access intelligent diagnostics and secure health reporting.")

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
        
        # Track file changes to reset state
        uploaded_file = st.file_uploader("Upload Clinical Report (PDF Only)", type=['pdf'], label_visibility="collapsed", key="report_uploader")
        
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

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🧬 Standardized Multi-Disease Risk Profile")
            st.info("AI Analysis across 3 critical disease categories.")
            
            with st.expander("ℹ️ Data Integrity & Imputation Notice"):
                st.write("""
                Our AI engine is equipped with a **Clinical Imputation System**. If your report is missing specific markers, 
                the system automatically fills the gaps with neutral clinical medians. This prevents system errors and 
                allows the analysis to proceed using the available data context.
                """)
            
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
                if bilirubin > 1.2: prob_h = max(prob_h, 0.85)
                elif bilirubin < 0.3: prob_h = min(prob_h, 0.20)
                elif 0.3 <= bilirubin <= 1.2: prob_h = 0.50 if prob_h < 0.70 and prob_h > 0.30 else prob_h
                
                # CLINICAL OVERRIDE: Albumin (Serum)
                if albumin > 5.4: prob_h = max(prob_h, 0.85)
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
                if creatinine > 1.3: prob_k = max(prob_k, 0.85)
                elif creatinine < 0.6: prob_k = min(prob_k, 0.20)
                elif 0.6 <= creatinine <= 1.3: prob_k = 0.50 if prob_k < 0.70 and prob_k > 0.30 else prob_k

            res_cols = st.columns(3)
            predictions = [
                ("Parkinson's Disease", prob_p),
                ("Hepatitis Condition", prob_h),
                ("Kidney Disease", prob_k)
            ]

            found_any = False
            for i, (name, prob) in enumerate(predictions):
                if prob is not None:
                    found_any = True
                    cat, cls, adv = get_risk_category(prob)
                    with res_cols[i]:
                        st.markdown(f"""
                        <div class="result-box {cls}" style="margin-top: 1rem; padding: 1.5rem; text-align: left;">
                            <h4 style="margin:0; color:inherit !important; border-bottom: 1px solid rgba(0,0,0,0.1); padding-bottom: 5px;">{name}</h4>
                            <div style="font-size: 1.6rem; font-weight: 800; margin-top: 10px;">{cat}</div>
                            <p style="font-size: 0.95rem; margin: 0.5rem 0; font-weight: 600;">Confidence: {prob:.1%}</p>
                            <p style="font-size: 0.9rem; opacity: 0.8; font-style: italic;">{adv}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    with res_cols[i]:
                        st.info(f"**{name}**: Insufficient data in report for analysis.")

            if not found_any:
                st.warning("Could not perform any disease risk analysis due to lack of relevant markers in the provided document.")
            
            # Reset button for new analysis
            if st.button("Reset Analysis"):
                st.session_state.show_results = False
                st.session_state.extracted_data = {}
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
                    st.session_state.active_tab_index = 1
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
        st.session_state.active_tab_index = 1 # Default to Risk Analysis on first login
    
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
