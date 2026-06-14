import os
import cv2
import numpy as np

IMAGE_SIZE = 128

def load_images(dataset_path):
    images = []
    labels = []

    classes = {
        "NonDemented": 0,
        "VeryMildDemented": 1,
        "MildDemented": 2,
        "ModerateDemented": 3
    }

    for class_name, label in classes.items():
        folder = os.path.join(dataset_path, class_name)

        if not os.path.exists(folder):
            print(f"Missing folder: {folder}")
            continue

        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)