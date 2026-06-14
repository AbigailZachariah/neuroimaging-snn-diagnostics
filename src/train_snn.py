import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump
from preprocesses import load_images
from src.encoder import rate_encode

os.makedirs("models", exist_ok=True)

LABELS = {
    0: "NonDemented",
    1: "VeryMildDemented",
    2: "MildDemented",
    3: "ModerateDemented"
}

print("Loading dataset...")
X, y = load_images("combined_images")
print("Dataset loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Normalize to 0-1
X = X.astype(np.float32) / 255.0

# --- Encode images into spike features using the SNN rate encoder ---
print("Encoding images into spike features (this may take a while)...")
X_encoded = []
for i, img in enumerate(X):
    spikes = rate_encode(img)            # (timesteps, 128, 128)
    features = spikes.sum(axis=0)        # (128, 128) — spike count per pixel across time
    X_encoded.append(features.flatten()) # 16384-dim vector
    if (i + 1) % 500 == 0:
        print(f"  Encoded {i + 1}/{len(X)} images")

X_encoded = np.array(X_encoded, dtype=np.float32)
print("Encoded shape:", X_encoded.shape)

# Save encoding stats for anomaly detection calibration
encoding_mean = float(X_encoded.mean())
encoding_std = float(X_encoded.std())
np.save("models/encoding_stats.npy", np.array([encoding_mean, encoding_std]))
print(f"Encoding stats — mean: {encoding_mean:.4f}, std: {encoding_std:.4f}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Train Random Forest on spike features
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
print("Training model...")
model.fit(X_train, y_train)

print("Evaluating...")
preds = model.predict(X_test)
print(classification_report(y_test, preds, target_names=list(LABELS.values())))

dump(model, "models/alzheimer_model.joblib")
print("Model saved to models/alzheimer_model.joblib")