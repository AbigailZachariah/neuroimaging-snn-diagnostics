from src.preprocesses import load_images
from src.predict import predict

images, labels = load_images("combined_images")

sample = images[0]

pred, score, anomaly = predict(sample)

print("Prediction:", pred)
print("Anomaly Score:", score)
print("Anomaly:", anomaly)