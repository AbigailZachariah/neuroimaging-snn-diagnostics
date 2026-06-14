import numpy as np
import joblib
from src.encoder import rate_encode
from src.anomaly import anomaly_score, is_anomaly

LABELS = {
    0: "NonDemented",
    1: "VeryMildDemented",
    2: "MildDemented",
    3: "ModerateDemented"
}

model = joblib.load("models/alzheimer_model.joblib")

try:
    stats = np.load("models/encoding_stats.npy")
    ENCODING_MEAN, ENCODING_STD = float(stats[0]), float(stats[1])
except FileNotFoundError:
    ENCODING_MEAN, ENCODING_STD = 5.0, 2.0
    print("Warning: encoding_stats.npy not found. Using default anomaly thresholds.")


def predict(image: np.ndarray) -> dict:
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    spikes = rate_encode(image)
    features = spikes.sum(axis=0)
    flat_features = features.flatten()

    prediction = model.predict(flat_features.reshape(1, -1))[0]
    label = LABELS[prediction]

    spike_mean = float(flat_features.mean())
    score = anomaly_score(spike_mean, ENCODING_MEAN, ENCODING_STD)
    flagged = is_anomaly(score)

    return {
        "prediction": int(prediction),
        "label": label,
        "anomaly_score": round(score, 4),
        "is_anomaly": flagged,
        "spike_mean": round(spike_mean, 4)
    }