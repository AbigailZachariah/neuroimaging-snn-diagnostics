from src.preprocesses import load_images
from src.encoder import rate_encode

images, labels = load_images("combined_images")

sample = images[0]

spikes = rate_encode(sample)

print("Original:", sample.shape)
print("Spikes:", spikes.shape)
print("Total spikes:", spikes.sum())