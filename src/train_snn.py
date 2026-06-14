import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

from preprocesses import load_images

print("Loading dataset...")

X, y = load_images("combined_images")

print("Dataset loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Normalize
X = X.astype(np.float32) / 255.0

# Flatten images
X = X.reshape(X.shape[0], -1)

print("Flattened shape:", X.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Simple baseline classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

print("Evaluating...")

preds = model.predict(X_test)

print(classification_report(y_test, preds))

dump(model, "models/alzheimer_model.joblib")

print("Model saved to models/alzheimer_model.joblib")