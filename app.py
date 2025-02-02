# app.py
import streamlit as st
import numpy as np
from models.sift_model import SIFTModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import cv2

# Initialize SIFT model
model = SIFTModel()

# Load and preprocess dataset
st.title("Low-Resolution Face Recognition using SIFT")
st.write("This app demonstrates face recognition on the ORL dataset using SIFT features.")

data_path = "dataset"
images, labels = model.load_orl_dataset(data_path)
processed_images = model.preprocess_dataset(images)

# Split dataset into training (HR) and testing (LR) sets
X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.4, random_state=42)

# Extract SIFT features
train_descriptors = model.extract_sift_features(X_train)
test_descriptors = model.extract_sift_features(X_test)

# Match descriptors and get predictions
predictions = model.match_descriptors(train_descriptors, test_descriptors, y_train)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
st.write(f"Recognition Accuracy: {accuracy * 100:.2f}%")

# Display test images and predictions
st.write("### Test Images and Predictions")
for i in range(len(X_test)):
    col1, col2 = st.columns(2)
    with col1:
        st.image(X_test[i], caption=f"Test Image {i+1}", use_column_width=True)
    with col2:
        st.write(f"Predicted Label: {predictions[i]}")