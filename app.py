# # app.py
# import streamlit as st
# import numpy as np
# from models.sift_model import SIFTModel
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Initialize SIFT model
# model = SIFTModel()

# # Load and preprocess dataset
# st.title("Low-Resolution Face Recognition using SIFT and HOG")
# st.write("This app demonstrates face recognition on the ORL dataset using SIFT features.")

# data_path = "data/orl_dataset"  # Ensure this points to the correct directory
# images, labels = model.load_orl_dataset(data_path)
# processed_images = model.preprocess_dataset(images)

# # Split dataset into training (HR) and testing (LR) sets
# X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.4, random_state=42)

# # Extract SIFT features
# train_descriptors = model.extract_sift_features(X_train)
# test_descriptors = model.extract_sift_features(X_test)

# # Match descriptors and get predictions
# predictions = model.match_descriptors(train_descriptors, test_descriptors, y_train)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, predictions)
# st.write(f"Recognition Accuracy: {accuracy * 100:.2f}%")

# # Display test images and predictions
# st.write("### Test Images and Predictions")
# for i in range(len(X_test)):
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(X_test[i], caption=f"Test Image {i+1}", use_column_width=True)
#     with col2:
#         st.write(f"Predicted Label: {predictions[i]}")





import streamlit as st
import numpy as np
from models.sift_model import FaceRecognitionModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize model
model = FaceRecognitionModel()

st.title("Low-Resolution Face Recognition using SIFT, HOG, and CNN")
st.write("This app demonstrates face recognition on the ORL dataset.")

# Load dataset
data_path = "data/orl_dataset"
images, labels = model.load_orl_dataset(data_path)
processed_images = model.preprocess_dataset(images)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.4, random_state=42)

# Model selection
model_choice = st.radio("Choose Model for Recognition:", ("SIFT", "HOG", "CNN"))

if model_choice == "SIFT":
    st.subheader("SIFT Feature Matching")
    train_descriptors = model.extract_sift_features(X_train)
    test_descriptors = model.extract_sift_features(X_test)
    predictions = model.match_descriptors(train_descriptors, test_descriptors, y_train)
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"SIFT Recognition Accuracy: {accuracy * 100:.2f}%")

elif model_choice == "HOG":
    st.subheader("HOG Feature Matching")
    train_hog_features = model.extract_hog_features(X_train)
    test_hog_features = model.extract_hog_features(X_test)
    
    # Use Nearest Neighbor for HOG
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_hog_features, y_train)
    predictions = knn.predict(test_hog_features)
    
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"HOG Recognition Accuracy: {accuracy * 100:.2f}%")

elif model_choice == "CNN":
    st.subheader("CNN Classification")
    if st.button("Train CNN Model"):
        model.train_cnn(X_train, y_train, X_test, y_test)
        st.success("CNN Model Trained!")

    if model.cnn_model:
        predictions = model.predict_cnn(X_test)
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"CNN Recognition Accuracy: {accuracy * 100:.2f}%")

# Display test images & predictions
st.write("### Test Images and Predictions")
for i in range(len(X_test)):
    col1, col2 = st.columns(2)
    with col1:
        st.image(X_test[i], caption=f"Test Image {i+1}", use_column_width=True)
    with col2:
        st.write(f"Predicted Label: {predictions[i]}")





