import joblib

from src.encoder import rate_encode
from src.anomaly import anomaly_score, is_anomaly

model = joblib.load("models/alzheimer_model.joblib")


def predict(image):
    spikes = rate_encode(image)

    features = spikes.sum(axis=0)

    prediction = model.predict(
        features.reshape(1, -1)
    )[0]

    score = anomaly_score(features)

    return prediction, score, is_anomaly(score)