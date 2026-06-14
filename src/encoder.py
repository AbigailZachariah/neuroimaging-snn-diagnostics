import numpy as np

def rate_encode(image, timesteps=10):
    """
    Convert grayscale image into spike trains.

    Input:
        image: 128x128 grayscale image

    Output:
        spikes: (timesteps, 128, 128)
    """

    image = image.astype(np.float32)

    if image.max() > 1:
        image = image / 255.0

    spikes = np.random.rand(
        timesteps,
        image.shape[0],
        image.shape[1]
    ) < image

    return spikes.astype(np.float32)