"""
Anomaly detection for the SNN inference pipeline.

Instead of hardcoded magic numbers, thresholds are derived from the
actual spike feature distribution computed during training and saved
to models/encoding_stats.npy.

A sample is flagged as anomalous if its mean spike activity deviates
more than Z_THRESHOLD standard deviations from the training mean.
This catches:
  - Non-MRI images fed to the model
  - Corrupted or blank scans
  - Images with very different contrast/brightness profiles
"""

# How many standard deviations away counts as anomalous.
# 2.5 keeps ~99% of real MRI scans as non-anomalous (tighter than 3-sigma).
Z_THRESHOLD = 2.5


def anomaly_score(spike_mean: float, training_mean: float, training_std: float) -> float:
    """
    Compute a z-score-based anomaly score.

    Args:
        spike_mean: mean spike activity of the current image's encoded features
        training_mean: mean spike activity seen across the training set
        training_std: std of spike activity across the training set

    Returns:
        float: z-score (how many std devs away from training distribution)
    """
    if training_std == 0:
        return 0.0
    return abs(spike_mean - training_mean) / training_std


def is_anomaly(score: float, threshold: float = Z_THRESHOLD) -> bool:
    """
    Returns True if the anomaly score exceeds the threshold.

    Args:
        score: z-score from anomaly_score()
        threshold: number of std deviations to flag as anomalous (default 2.5)

    Returns:
        bool
    """
    return score > threshold