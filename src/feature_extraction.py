import cv2
import numpy as np
from skimage.feature import hog

def extract_features(image_path, image_size=(72, 72)):
    """Extract HOG features from an image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, image_size)  # Resize image to 72x72
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features
