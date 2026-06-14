import numpy as np

THRESHOLD = 8.0


def anomaly_score(features):
    mean_activity = np.mean(features)

    expected = 10.0

    return abs(mean_activity - expected)


def is_anomaly(score):
    return score > THRESHOLD