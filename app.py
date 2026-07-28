import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Stroke Risk Predictor | Clinical AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS — Professional Medical Theme
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    /* Hero header */
    .hero-container {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 40px rgba(14, 165, 233, 0.25);
    }
    .hero-title {
        color: white;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: 0.4rem;
    }

    /* Card containers */
    .info-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem 1.7rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        color: #e2e8f0;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Metric-style result cards */
    .result-card {
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-high {
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%);
        border: 1px solid #ef4444;
    }
    .result-low {
        background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        border: 1px solid #10b981;
    }
    .result-label {
        color: rgba(255,255,255,0.85);
        font-size: 0.95rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-value {
        color: white;
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b1120;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 0;
        transition: all 0.25s ease;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 28px rgba(99, 102, 241, 0.5);
    }

    /* Inputs */
    div[data-baseweb="select"] > div, .stNumberInput input {
        background-color: rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-blue { background: rgba(14,165,233,0.15); color: #38bdf8; border: 1px solid rgba(14,165,233,0.3); }
    .badge-purple { background: rgba(99,102,241,0.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.3); }

    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL (cached so it isn't reloaded on every interaction)
# =========================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("stroke_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# =========================================================
# SESSION STATE — history of predictions this session
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# HERO HEADER
# =========================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧠 Stroke Risk Prediction System</div>
    <div class="hero-subtitle">AI-assisted clinical decision support powered by a Random Forest classifier</div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠ Could not load model artifacts. Make sure `stroke_model.pkl` and `scaler.pkl` are in the app directory.\n\nDetails: {load_error}")
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🩺 About this tool")
    st.info(
        "This application estimates an individual's probability of stroke "
        "using a machine learning model trained on clinical and lifestyle data.\n\n"
        "**Model:** Random Forest Classifier\n\n"
        "**Inputs:** Demographics, medical history, and lifestyle factors."
    )

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    st.metric("Predictions made", len(st.session_state.history))
    if st.session_state.history:
        avg_risk = np.mean([h["probability"] for h in st.session_state.history])
        st.metric("Average predicted risk", f"{avg_risk*100:.1f}%")

    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.caption(
        "This tool is for educational and research purposes only. "
        "It is **not** a substitute for professional medical diagnosis. "
        "Always consult a qualified healthcare provider."
    )

    st.markdown("---")
    st.caption(f"Session started: {datetime.now().strftime('%d %b %Y, %H:%M')}")

# =========================================================
# INPUT FORM
# =========================================================
tab_predict, tab_history = st.tabs(["🔍 Prediction", "📜 History"])

with tab_predict:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)

        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", 1, 100, 30)
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
        work_type = st.selectbox(
            "Work Type",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
        )
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🩺 Medical & Lifestyle</div>', unsafe_allow_html=True)

        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        glucose = st.number_input("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        smoking = st.selectbox("Smoking Status", ["Never Smoked", "Formerly Smoked", "Smokes", "Unknown"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍  Predict Stroke Risk", use_container_width=True)

    # -----------------------------
    # Encoding
    # -----------------------------
    gender_enc = 1 if gender == "Male" else 0
    hypertension_enc = 1 if hypertension == "Yes" else 0
    heart_disease_enc = 1 if heart_disease == "Yes" else 0
    ever_married_enc = 1 if ever_married == "Yes" else 0
    residence_enc = 1 if residence == "Urban" else 0

    work_dict = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}
    smoke_dict = {"Never Smoked": 0, "Formerly Smoked": 1, "Smokes": 2, "Unknown": 3}

    input_data = pd.DataFrame([[
        gender_enc, age, hypertension_enc, heart_disease_enc, ever_married_enc,
        work_dict[work_type], residence_enc, glucose, bmi, smoke_dict[smoking]
    ]], columns=[
        "gender", "age", "hypertension", "heart_disease", "ever_married",
        "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"
    ])

    # -----------------------------
    # Prediction
    # -----------------------------
    if predict_clicked:
        with st.spinner("Analyzing patient data..."):
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)
            probability = float(model.predict_proba(scaled_data)[0][1])

        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "age": age,
            "gender": gender,
            "probability": probability,
            "risk": "High" if prediction[0] == 1 else "Low"
        })

        st.markdown("---")
        st.markdown('<div class="section-title">📈 Prediction Result</div>', unsafe_allow_html=True)

        result_col, gauge_col = st.columns([1, 1.3], gap="large")

        with result_col:
            if prediction[0] == 1:
                st.markdown(f"""
                <div class="result-card result-high">
                    <div class="result-label">⚠ High Risk Detected</div>
                    <div class="result-value">{probability*100:.1f}%</div>
                    <div style="color:rgba(255,255,255,0.8); font-size:0.9rem;">Estimated stroke probability</div>
                </div>
                """, unsafe_allow_html=True)
                st.warning("Please consult a healthcare professional for further evaluation and guidance.")
            else:
                st.markdown(f"""
                <div class="result-card result-low">
                    <div class="result-label">✅ Low Risk</div>
                    <div class="result-value">{probability*100:.1f}%</div>
                    <div style="color:rgba(255,255,255,0.8); font-size:0.9rem;">Estimated stroke probability</div>
                </div>
                """, unsafe_allow_html=True)
                st.success("Continue maintaining a healthy lifestyle and routine checkups.")

        with gauge_col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "font": {"size": 40, "color": "white"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white", "tickfont": {"color": "white"}},
                    "bar": {"color": "#6366f1"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "#059669"},
                        {"range": [30, 60], "color": "#d97706"},
                        {"range": [60, 100], "color": "#dc2626"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": probability * 100
                    }
                },
                domain={"x": [0, 1], "y": [0, 1]}
            ))
            fig.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="badge badge-blue">Random Forest Classifier</div>'
                     '<div class="badge badge-purple">Scaled Input Features</div>', unsafe_allow_html=True)

        with st.expander("📋 View Patient Input Data"):
            st.dataframe(input_data, use_container_width=True)

with tab_history:
    st.markdown('<div class="section-title">📜 Prediction History (this session)</div>', unsafe_allow_html=True)
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df["probability"] = (hist_df["probability"] * 100).round(2).astype(str) + "%"
        st.dataframe(hist_df, use_container_width=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions made yet in this session. Head to the Prediction tab to get started.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer-text">
    Built with Streamlit • Scikit-learn Random Forest • Plotly<br>
    ⚠ For educational purposes only — not a medical diagnostic tool.
</div>
""", unsafe_allow_html=True)
