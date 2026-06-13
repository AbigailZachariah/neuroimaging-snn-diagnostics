import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="SynapseAnomalX", page_icon="🧠", layout="wide")

st.title("🧠 SynapseAnomalX")
st.subheader("Brain-Inspired Anomaly Detection for Cognitive Decline")
st.markdown("---")

uploaded = st.file_uploader("Upload MRI slice (.npy file)", type=['npy'])

if uploaded:
    data = np.load(uploaded, allow_pickle=True).item()
    slc = data['slice']

    st.image(slc, caption="MRI Slice", clamp=True)
    st.info("⏳ Model loading... (integration coming soon)")
else:
    st.warning("👆 Please upload an MRI slice file to begin analysis")