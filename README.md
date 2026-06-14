# 🧠 SynapseAnomalX: Brain-Inspired Anomaly Detection

SynapseAnomalX is an advanced neuroimaging diagnostic dashboard that leverages Spiking Neural Networks (SNNs) and unsupervised feature extraction to detect early-stage cognitive decline patterns from MRI structural scans.

---

## 🚀 Core Innovation & Tech Stack

* *Biologically Inspired Architecture:* Uses Spiking Neural Networks (SNN) combined with STDP-inspired behavior to trace structural deviation.
* *Unsupervised Anomaly Scoring:* Trained exclusively on healthy brain data (~35 scans) to learn normal baseline pathways, utilizing Z-Score thresholds ($Z = 2.5$) to isolate anomalous cognitive decline.
* *Dynamic Analytics Engine:* Processes incoming gray-scale MRI slices on-the-fly and translates spike train frequencies into dynamic frontend warning states.

---

## 📂 Project Repository Structure

* src/ — Core algorithmic backend execution pipeline:
  * predict.py — Pipeline coordinator loading model weights and returning calculated anomaly predictions.
  * anomaly.py — Mathematical evaluation script checking features against $Z$-threshold limits.
  * dataset.py — Lazy loading PyTorch dataset component with live image augmentation handlers.
* frontend/ — System presentation layout:
  * app.py — Deeply styled Streamlit interactive frontend dashboard reading from the src inference module.
* models/ — Pre-trained parameters, scaling coefficients, and statistical arrays (alzheimer_model.joblib).

---

## 🛠️ Installation & Local Execution

Follow these step-by-step instructions to boot up the diagnostics workspace locally:

### 1. Set Up Environment & Dependencies
Ensure your python virtual workspace (venv) is activated, then run:
```bash
pip install streamlit numpy matplotlib scikit-learn joblib opencv-python pillow torch
