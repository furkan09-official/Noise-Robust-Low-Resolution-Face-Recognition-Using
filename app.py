
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from models.combined_model import FaceRecognitionModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize model
model = FaceRecognitionModel()

st.set_page_config(layout="wide")
st.title("Noise Robust Low-Resolution Face Recognition using SIFT, HOG And CNN")
st.markdown("This app demonstrates face recognition on the **ORL dataset** using different models.")

# Load dataset
data_path = "data/orl_dataset"
images, labels = model.load_orl_dataset(data_path)
processed_images = model.preprocess_dataset(images)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.2, random_state=42)

# Model selection
model_choice = st.radio("🔍 Choose Model for Recognition:", ("SIFT", "HOG", "CNN","SIFT + CNN"))

predictions = []
accuracy = 0.0

def show_speedometer(accuracy):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=accuracy * 100,
        title={'text': "Recognition Accuracy (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightcoral"},
                {'range': [50, 75], 'color': "khaki"},
                {'range': [75, 100], 'color': "lightgreen"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)





def display_predictions(X_test, y_test, predictions):
    st.subheader("📷 Test Images")
    min_len = min(len(X_test), len(predictions), len(y_test))
    for i in range(0, min_len, 3):
        cols = st.columns(3)
        for j in range(3):
            idx = i + j
            if idx < min_len:
                with cols[j]:
                    st.image(X_test[idx], caption=f"Pred: {predictions[idx]} | True: {y_test[idx]}")
                    correct = predictions[idx] == y_test[idx]
                    st.markdown(
                        f"<span style='color: {'green' if correct else 'red'};'>✔ Correct</span>" if correct else
                        f"<span style='color: red;'>✖ Wrong</span>", unsafe_allow_html=True
                    )







if model_choice == "SIFT":
    st.subheader("🔑 SIFT Feature Matching")
    train_descriptors = model.extract_sift_features(X_train)
    test_descriptors = model.extract_sift_features(X_test)
    predictions = model.match_descriptors(train_descriptors, test_descriptors, y_train)
    accuracy = accuracy_score(y_test, predictions)
    # show_speedometer(accuracy)
    show_speedometer(accuracy)
    display_predictions(X_test, y_test, predictions)


elif model_choice == "HOG":
    st.subheader("📐 HOG Feature Matching")
    train_hog_features = model.extract_hog_features(X_train)
    test_hog_features = model.extract_hog_features(X_test)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_hog_features, y_train)
    predictions = knn.predict(test_hog_features)

    accuracy = accuracy_score(y_test, predictions)
    # show_speedometer(accuracy)
    show_speedometer(accuracy)
    display_predictions(X_test, y_test, predictions)


elif model_choice == "CNN":
    st.subheader("🧠 CNN Classification")
    if st.button("🚀 Train CNN Model"):
        model.train_cnn(X_train, y_train, X_test, y_test)
        st.success("✅ CNN Model Trained!")

    if model.cnn_model:
        predictions = model.predict_cnn(X_test)
        accuracy = accuracy_score(y_test, predictions)
        show_speedometer(accuracy)

        # Per-image prediction and accuracy
        st.subheader("📊 Individual Predictions")
        for i in range(0, len(X_test), 3):
            cols = st.columns(3)
            for j in range(3):
                idx = i + j
                if idx < len(X_test):
                    with cols[j]:
                        st.image(X_test[idx], caption=f"Pred: {predictions[idx]} | True: {y_test[idx]}")
                        correct = predictions[idx] == y_test[idx]
                        st.markdown(
                            f"<span style='color: {'green' if correct else 'red'};'>✔ Correct</span>" if correct else
                            f"<span style='color: red;'>✖ Wrong</span>", unsafe_allow_html=True
                        )
elif model_choice == "SIFT + HOG":
    st.subheader("🔬 Combined SIFT + HOG Feature Matching")
    train_features = model.extract_sift_hog_features(X_train)
    test_features = model.extract_sift_hog_features(X_test)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_features, y_train)
    predictions = knn.predict(test_features)

    accuracy = accuracy_score(y_test, predictions)
    show_speedometer(accuracy)


elif model_choice == "SIFT + CNN":
    st.subheader("🤖 Combined SIFT + CNN Model")

    # Extract SIFT + HOG features for train and test (as input to CNN)
    train_sift_features = model.extract_sift_hog_features(X_train)
    test_sift_features = model.extract_sift_hog_features(X_test)

    # Train combined model using images + sift features
    model.train_sift_cnn(X_train, train_sift_features, y_train, X_test, test_sift_features, y_test)

    # Predict on test data
    predictions = model.predict_sift_cnn(X_test, test_sift_features)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Show accuracy with your custom speedometer function
    # show_speedometer(accuracy)
    show_speedometer(accuracy)
    display_predictions(X_test, y_test, predictions)



# Optional: show full results in expandable section
with st.expander("📋 Show All Predictions"):
    min_len = min(len(X_test), len(predictions), len(y_test))
    for i in range(len(X_test)):
        st.image(X_test[i], caption=f"Test Image {i+1} | True: {y_test[i]}", width=150)

