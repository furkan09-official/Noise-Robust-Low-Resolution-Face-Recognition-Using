import streamlit as st
import cv2
import numpy as np
import joblib
from feature_extraction import extract_features

# Load models
models = {
    "SVM": joblib.load("models/SVM.pkl"),
    "Random Forest": joblib.load("models/Random Forest.pkl"),
    "KNN": joblib.load("models/KNN.pkl")
}

st.title("Face Recognition with ORL Dataset")

uploaded_file = st.file_uploader("Upload a Face Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract features
    features = extract_features(image).reshape(1, -1)

    # Predictions
    for name, model in models.items():
        pred = model.predict(features)[0]
        st.write(f"**{name} Prediction:** Subject {pred}")
