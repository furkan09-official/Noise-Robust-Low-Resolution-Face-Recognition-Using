import joblib
import cv2
import numpy as np
from feature_extraction import extract_features

def predict(image_path, model_name="SVM"):
    """Predict the class of a given image using the specified model."""
    model = joblib.load(f"models/{model_name}.pkl")
    features = extract_features(image_path).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

if __name__ == "__main__":
    test_image = "dataset/subject1/test1.jpg"
    print("Predicted Class:", predict(test_image))
