from src.preprocesses import load_images
import numpy as np

images, labels = load_images("combined_images")

print("Images:", images.shape)
print("Labels:", labels.shape)

print("Unique labels:", np.unique(labels))
print("Label counts:")

for label in np.unique(labels):
    print(label, np.sum(labels == label))