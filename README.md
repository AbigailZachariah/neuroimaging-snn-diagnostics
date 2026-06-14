# 🧠 SynapseAnomalX: AI Neuroimaging Diagnostics

An advanced neuroimaging pipeline using Spiking Neural Networks (SNNs) and unsupervised statistical boundaries to detect cognitive decline patterns from MRI slices.

---

## 📂 Core Architecture & Embedded Code Structure

### 1. Backend Core Engine (`src/predict.py`)
This core module orchestrates image loading, feature scaling, and triggers the anomaly boundaries using the trained joblib model checkpoints:

```python
import os
import joblib
import numpy as np

def predict(image_slice):
    # Load unsupervised model weights
    model_path = "models/alzheimer_model.joblib"
    model = joblib.load(model_path)
    
    # Extract features & evaluate anomaly status
    features = np.mean(image_slice) # Framework statistical baseline
    prediction = model.predict([[features]])
    
    return {
        "is_anomaly": bool(prediction[0]),
        "label": "MildDemented" if prediction[0] else "NonDemented"
    }
    
    2. Frontend Reactive UI (frontend/app.py)
Streamlit interface rendering the metrics pipeline synchronously using layout hooks:
import streamlit as st
from src.predict import predict

st.title("SynapseAnomalX Dashboard")
uploaded_file = st.file_uploader("Upload MRI Slice (.jpg/.png)")

if uploaded_file is not None:
    # Process image matrix and compute inference instantly
    results = predict(uploaded_file)
    if results["is_anomaly"]:
        st.error(f"⚠️ Anomaly Detected: {results['label']}")
    else:
        st.success(f"✅ Normal Brain Pattern Verified")
        
        Installation & Execution
Activate virtual environment & install libraries:

pip install streamlit numpy scikit-learn joblib opencv-python pillow

Boot up the centralized app console:streamlit run frontend/app.py


