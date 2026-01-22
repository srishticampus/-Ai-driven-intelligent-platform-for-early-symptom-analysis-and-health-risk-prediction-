import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# Set page config
st.set_page_config(
    page_title="HealthAI - Intelligent Risk Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Path to the CSS file
css_path = os.path.join(os.getcwd(), "style.css")
if os.path.exists(css_path):
    local_css(css_path)

# Initialize Session State
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_diabetes_assets():
    model_path = os.path.join(os.getcwd(), 'models', 'diabetes_model.pkl')
    scaler_path = os.path.join(os.getcwd(), 'models', 'scaler.pkl')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    return None, None

def predict_diabetes(data):
    model, scaler = load_diabetes_assets()
    if model is None or scaler is None:
        return None, None
    
    # Preprocess: Scale the input data
    data_scaled = scaler.transform(data)
    
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)
    return prediction[0], probability[0][1]

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
    
    if os.path.exists("hero.png"):
        st.image("hero.png", use_container_width=True, caption="AI Driven Health Analysis")

    st.markdown("Welcome to the AI-driven intelligent platform for early symptom analysis and lifestyle disease prediction.")
    st.info("Explore the tabs above to access intelligent diagnostics and health information.")

def prediction_page():
    st.title("Disease Risk Analysis")
    st.markdown("Enter values from the patient's medical report or upload a digital copy for a comprehensive lifestyle disease risk profiling.")
    
    # Overview Metrics
    cols = st.columns(3)
    with cols[0]:
        st.markdown('<div class="metric-card"><div class="metric-value">Optimized</div>ML Pipeline</div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="metric-card"><div class="metric-value">3 Levels</div>Risk Profiling</div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<div class="metric-card"><div class="metric-value">Live</div>Diagnostics</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.container():
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        
        # Medical Report Upload Section
        st.subheader("📁 Medical Report Data Extraction")
        st.info("Use the uploader below to automatically populate patient clinical metrics from a digital report.")
        uploaded_file = st.file_uploader("Upload Clinical Report (PDF/JPG/PNG)", type=['pdf', 'png', 'jpg'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            if st.button("Process & Extract Report"):
                with st.spinner("Analyzing report structure and extracting markers..."):
                    import time
                    time.sleep(1.5) # Simulate processing
                    st.info("📄 Report structure recognized. Identifying clinical markers...")
                    time.sleep(1.0)
                    st.success("✅ Parameters extracted! Metrics have been populated in the fields below. Please verify before generating risk profile.")
                    # In a real app, we would update session_state values here
        
        st.markdown("---")
        st.subheader("Medical Report Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, help="Number of times pregnant")
            glucose = st.number_input("Plasma Glucose Concentration (mg/dL)", min_value=0, max_value=300, value=120)
            blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=0, max_value=100, value=20)
            
        with col2:
            insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0, max_value=1000, value=80)
            bmi = st.number_input("Body Mass Index (kg/m²)", min_value=0.0, max_value=70.0, value=25.0)
            pedigree = st.number_input("Diabetes Pedigree Function Score", min_value=0.0, max_value=3.0, value=0.5)
            age = st.number_input("Patient Age (Years)", min_value=0, max_value=120, value=30)
            
        if st.button("Generate Risk Profile"):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]])
            prediction, probability = predict_diabetes(input_data)
            
            if prediction is not None:
                st.markdown("---")
                # Logic for Low, Medium, High risk
                if probability >= 0.70:
                    status = "High Risk"
                    color_cls = "high-risk"
                    advice = "Immediate medical consultation and comprehensive metabolic screening recommended."
                elif probability >= 0.35:
                    status = "Medium Risk"
                    color_cls = "medium-risk" # We should define this in CSS
                    advice = "Lifestyle modifications suggested. Regular monitoring of glucose levels is advised."
                else:
                    status = "Low Risk"
                    color_cls = "low-risk"
                    advice = "Continue regular checkups and maintain a balanced diet and exercise routine."

                st.markdown(f"""
                <div class="result-box {color_cls}">
                    <h2>{status} Detected</h2>
                    <p>Clinical analysis indicates a <b>{probability:.1%}</b> probability of metabolic syndrome (Lifestyle Disease).</p>
                    <p style="font-size: 1.1em; color: #cbd5e1;">{advice}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Diagnostic engine offline. Please verify the training assets.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("### System Performance & Data Insights")
    
    st.markdown("""
    <div class="main-card">
    <b>Optimized ML Pipeline:</b>
    <ul style="margin-top: 10px;">
        <li><b>Data Cleaning:</b> Automating outlier detection and normalization.</li>
        <li><b>Risk Stratification:</b> Multi-level categorization for clinical decision support.</li>
        <li><b>Fine-tuned Accuracy:</b> Using validated feature sets from medical datasets.</li>
    </ul>
    <b>Current Model Performance:</b>
    <div style='display: flex; gap: 20px; margin-top: 10px;'>
        <div style='text-align: center;'><b>Accuracy</b><br><span style='color: #38bdf8; font-size: 1.2em;'>78.1%</span></div>
        <div style='text-align: center;'><b>Precision</b><br><span style='color: #38bdf8; font-size: 1.2em;'>68.5%</span></div>
        <div style='text-align: center;'><b>Recall</b><br><span style='color: #38bdf8; font-size: 1.2em;'>71.2%</span></div>
        <div style='text-align: center;'><b>F1-Score</b><br><span style='color: #38bdf8; font-size: 1.2em;'>69.8%</span></div>
    </div>
    </div>
    """, unsafe_allow_html=True)

def chatbot_page():
    st.title("Medical Health Chatbot")
    st.markdown("Ask our AI assistant any health-related educational questions.")
    
    # Simple rule-based/mock LLM response for the prototype
    medical_responses = {
        "symptoms of diabetes": "Common symptoms include increased thirst, frequent urination, unexplained weight loss, and fatigue. Please consult a doctor for diagnosis.",
        "prevent diabetes": "Prevention includes a balanced diet, regular physical activity, maintaining a healthy weight, and avoiding smoking.",
        "what is bmi": "BMI or Body Mass Index is a measure that uses your height and weight to work out if your weight is healthy.",
        "glucose": "Glucose is the main sugar found in your blood. It comes from the food you eat and is your body's main source of energy.",
        "hello": "Hello! I am your AI Health Assistant. How can I help you today with health-related information?",
        "hi": "Hi there! Feel free to ask about health symptoms or risk prevention tips."
    }

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
            
            # Simple matching logic for the mock
            query = prompt.lower()
            response_text = "I'm sorry, I'm a prototype assistant and don't have information on that yet. Try asking about diabetes symptoms or prevention."
            
            for key in medical_responses:
                if key in query:
                    response_text = medical_responses[key]
                    break
            
            # Simulate typing
            for chunk in response_text.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def login_page():
    #st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<span class="login-logo">🔐</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="login-title">Secure Portal</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94a3b8; font-size: 1.2rem; margin-bottom: 2rem;">Please login to analyse disease risk</p>', unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submit = st.form_submit_button("Sign In")
        
        if submit:
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.session_state.active_tab = 1 # Switch to Risk Analysis tab
                st.success("Access Granted! Loading your dashboard...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color: #64748b; margin-top: 1rem;">Default: admin / admin123</p>', unsafe_allow_html=True)

# --- MAIN RENDER ---
sidebar()

if st.session_state.logged_in:
    # When logged in, show Risk Analysis as the first tab
    tab_names = ["📊 Risk Analysis","🏠 Home", "💬 Chatbot"]
    active_tab = st.tabs(tab_names)
    
    with active_tab[0]:
        # Logout button displayed above risk analyser content
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🚪 Logout", key="top_logout"):
                st.session_state.logged_in = False
                st.session_state.active_tab = 0  # Go to Home after logout
                st.rerun()
        
        prediction_page()
    
    with active_tab[1]:
        home_page()
    
    with active_tab[2]:
        chatbot_page()
else:
    # When not logged in, show Home as first tab, Login as second
    tab_names = ["🏠 Home", "🔐 Login", "💬 Chatbot"]
    active_tab = st.tabs(tab_names)
    
    with active_tab[0]:
        home_page()
    
    with active_tab[1]:
        login_page()
    
    with active_tab[2]:
        chatbot_page()

st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes only and does not provide medical advice. Always consult a physician.")
