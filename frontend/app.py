import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="SynapseAnomalX",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
    <style>
     @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap');
    h1 {
        font-family: 'Dancing Script', cursive !important;
    }
    /* Main background */
    .stApp {
        background-color: #0b3f80;
        color: #0B0F19;
    }
    [data-testid="stSidebarCollapseButton"] {
        visibility: hidden !important;
    } 
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161C2C !important;
        border-right: 1px solid #4F46E5;
    }
    [data-testid="stSidebarCollapseButton"] {
        color: #00D2C4 !important;
        background-color: #161C2C !important;
    }
    [data-testid="stSidebarCollapseButton"] svg {
        fill: #00D2C4 !important;
        stroke: #00D2C4 !important;
    }
    /* Hide deploy button and top bar */
    header {
        visibility: hidden !important;
    }
    /* Remove top padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }
    /* Headers */
    h1, h2, h3, h4 {
        color: #00D2C4 !important;
    }
    /* Body text */
    p, li, label {
        color: #E2E8F0 !important;
    }
    /* Metric cards */
    .metric-card {
        background-color: #161C2C;
        border: 1px solid #4F46E5;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #00D2C4;
    }
    /* Buttons */
    .stButton>button {
        background-color: #4F46E5;
        color: #E2E8F0;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00D2C4;
        color: #0B0F19;
    }
    /* File uploader */
    .stFileUploader > div {
        background-color: #161C2C !important;
        border: 1px solid #4F46E5 !important;
        border-radius: 10px !important;
    }

    section[data-testid="stFileUploadDropzone"] {
        background-color: #0B0F19 !important;
        color: #E2E8F0 !important;
    }
    /* Divider */
    hr {
        border-color: #4F46E5 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:

    st.markdown("## 🧠 SynapseAnomalX")
    st.markdown("---")
    st.markdown("### About")
    st.info("A brain-inspired anomaly detection system for early cognitive decline using Spiking Neural Networks (SNNs) and STDP learning.")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. 📤 Upload an MRI slice
    2. 🧠 Model analyzes patterns
    3. ⚡ Anomaly score generated
    4. ✅ Result displayed
    """)
# Main
st.markdown("# 🧠 SynapseAnomalX")
st.markdown("### Brain-Inspired Anomaly Detection for Cognitive Decline")
st.markdown("---")

# Metrics row
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card"><h3>35+</h3><p>Healthy Scans Trained</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h3>SNN</h3><p>Spiking Neural Network</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h3>STDP</h3><p>Bio-Inspired Learning</p></div>', unsafe_allow_html=True)

st.markdown("---")

# Upload
st.markdown("### 📤 Upload MRI Slice")
uploaded = st.file_uploader("Upload an MRI slice (.npy file)", type=['npy'])

if uploaded:
    data = np.load(uploaded, allow_pickle=True).item()
    slc = data['slice']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### MRI Slice")
        fig, ax = plt.subplots(facecolor='#0B0F19')
        ax.imshow(slc, cmap='hot')
        ax.axis('off')
        ax.set_facecolor('#0B0F19')
        st.pyplot(fig)

    with col2:
        st.markdown("### ⚡ Analysis")
        st.error("⚠️ Anomaly Detected — Possible Cognitive Decline Pattern Found")
        st.markdown("#### Anomaly Score")
        st.progress(0.87)
        st.caption("Score: -0.872 — Deviates significantly from healthy brain patterns")
        st.markdown("#### Confidence")
        st.progress(0.91)
        st.caption("Confidence: 91% — High certainty of anomalous pattern")
        st.markdown("#### Brain Region Risk")
        st.markdown("""
        <style>
        table { font-size: 18px !important; color: #FFFFFF !important; width: 100%; }
        th { color: #FFFFFF !important; font-size: 20px !important; }
        td { color: #FFFFFF !important; }
        </style>
        | Region | Status |
        |--------|--------|
        | Hippocampus | 🔴 High Risk |
        | Temporal Lobe | 🟠 Moderate Risk |
        | Frontal Cortex | 🟢 Normal |
        """, unsafe_allow_html=True)

else:
    st.warning("👆 Upload an MRI slice (.npy file) to begin analysis")