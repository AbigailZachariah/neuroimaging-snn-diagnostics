import numpy as np

Z_THRESHOLD = 2.5

def anomaly_score(spike_mean: float, training_mean: float, training_std: float) -> float:
    if training_std == 0:
        return 0.0
    return abs(spike_mean - training_mean) / training_std

def is_anomaly(score: float, threshold: float = Z_THRESHOLD) -> bool:
    return score > threshold