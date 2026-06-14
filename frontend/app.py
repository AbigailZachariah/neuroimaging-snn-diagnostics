import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure the root directory is in python path so it can read 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict

st.set_page_config(
    page_title="SynapseAnomalX",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling Layout
st.markdown("""
    <style>
     @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap');
    h1 {
        font-family: 'Dancing Script', cursive !important;
    }
    .stApp {
        background-color: #0b3f80;
        color: #0B0F19;
    }
    [data-testid="stSidebarCollapseButton"] {
        visibility: hidden !important;
    } 
    [data-testid="stSidebar"] {
        background-color: #161C2C !important;
        border-right: 1px solid #4F46E5;
    }
    header {
        visibility: hidden !important;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }
    h1, h2, h3, h4 {
        color: #00D2C4 !important;
    }
    p, li, label {
        color: #E2E8F0 !important;
    }
    .metric-card {
        background-color: #161C2C;
        border: 1px solid #4F46E5;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #00D2C4;
    }
    .stFileUploader > div {
        background-color: #161C2C !important;
        border: 1px solid #4F46E5 !important;
        border-radius: 10px !important;
    }
    section[data-testid="stFileUploadDropzone"] {
        background-color: #0B0F19 !important;
        color: #E2E8F0 !important;
    }
    hr {
        border-color: #4F46E5 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar UI
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

# Main Title Headers
st.markdown("# 🧠 SynapseAnomalX")
st.markdown("### Brain-Inspired Anomaly Detection for Cognitive Decline")
st.markdown("---")

# Global Metrics Summary Cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card"><h3>35+</h3><p>Healthy Scans Trained</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h3>SNN</h3><p>Spiking Neural Network</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h3>STDP</h3><p>Bio-Inspired Learning</p></div>', unsafe_allow_html=True)

st.markdown("---")

# File Upload Processing Section
st.markdown("### 📤 Upload MRI Slice")
uploaded = st.file_uploader("Upload an MRI scan (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded:
    from PIL import Image
    # Process image to match exactly what src/predict.py expects
    img_pil = Image.open(uploaded).convert('L').resize((128, 128))
    slc = np.array(img_pil).astype(np.float32) / 255.0

    # Run actual model prediction pipeline
    results = predict(slc)
    
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
        
        # Display dynamically based on whether actual model flagged an anomaly
        if results["is_anomaly"]:
            st.error(f"⚠️ Anomaly Detected — Possible {results['label']} Pattern Found")
        else:
            st.success(f"✅ Normal Brain Pattern Verified ({results['label']})")
            
        st.markdown("#### Anomaly Deviation Metric")
        # Normalize score between 0.0 and 1.0 for the progress bar rendering safely
        progress_val = min(max(float(results["anomaly_score"]) / 5.0, 0.0), 1.0)
        st.progress(progress_val)
        st.caption(f"Calculated Z-Score: {results['anomaly_score']} — Spike Mean: {results['spike_mean']}")
        
        st.markdown("#### Diagnostic Status Classification")
        st.info(f"Target Diagnostic Category: **{results['label']}** (Class Code: {results['prediction']})")
        
        st.markdown("#### Brain Region Vulnerability Risk")
        # Dynamically color map risk categories based on classification
        hip_status = "🔴 High Risk" if results["prediction"] >= 2 else "🟠 Moderate Risk" if results["prediction"] == 1 else "🟢 Normal"
        tem_status = "🔴 High Risk" if results["prediction"] == 3 else "🟠 Moderate Risk" if results["prediction"] >= 1 else "🟢 Normal"
        
        st.markdown(f"""
        <style>
        table {{ font-size: 18px !important; color: #FFFFFF !important; width: 100%; }}
        th {{ color: #FFFFFF !important; font-size: 20px !important; }}
        td {{ color: #FFFFFF !important; }}
        </style>
        | Region | Status |
        |--------|--------|
        | Hippocampus | {hip_status} |
        | Temporal Lobe | {tem_status} |
        | Frontal Cortex | 🟢 Normal |
        """, unsafe_allow_html=True)

else:
    st.warning("👆 Upload an MRI slice (.jpg/.png file) to begin analysis")