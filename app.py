
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from models.sift_model import FaceRecognitionModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize model
model = FaceRecognitionModel()

st.set_page_config(layout="wide")
st.title("🧠 Low-Resolution Face Recognition using SIFT, HOG and CNN and sift + hog")
st.markdown("This app demonstrates face recognition on the **ORL dataset** using different models.")

# Load dataset
data_path = "data/orl_dataset"
images, labels = model.load_orl_dataset(data_path)
processed_images = model.preprocess_dataset(images)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.2, random_state=42)

# Model selection
model_choice = st.radio("🔍 Choose Model for Recognition:", ("SIFT", "HOG", "CNN","SIFT + HOG"))

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

if model_choice == "SIFT":
    st.subheader("🔑 SIFT Feature Matching")
    train_descriptors = model.extract_sift_features(X_train)
    test_descriptors = model.extract_sift_features(X_test)
    predictions = model.match_descriptors(train_descriptors, test_descriptors, y_train)
    accuracy = accuracy_score(y_test, predictions)
    show_speedometer(accuracy)

elif model_choice == "HOG":
    st.subheader("📐 HOG Feature Matching")
    train_hog_features = model.extract_hog_features(X_train)
    test_hog_features = model.extract_hog_features(X_test)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_hog_features, y_train)
    predictions = knn.predict(test_hog_features)

    accuracy = accuracy_score(y_test, predictions)
    show_speedometer(accuracy)

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
                        st.image(X_test[idx], caption=f"Pred: {predictions[idx]} | True: {y_test[idx]}", use_container_width=True)
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


# Optional: show full results in expandable section
with st.expander("📋 Show All Predictions"):
    for i in range(len(X_test)):
        st.image(X_test[i], caption=f"Test Image {i+1} | Predicted: {predictions[i]} | True: {y_test[i]}", width=150)


# import streamlit as st
# import numpy as np
# import plotly.graph_objects as go
# from models.sift_model import FaceRecognitionModel
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Initialize model
# model = FaceRecognitionModel()

# st.set_page_config(layout="wide")
# st.title("🧠 Low-Resolution Face Recognition")
# st.markdown("""
# ### 👁️‍🗨️ Using SIFT, HOG, and CNN on ORL Face Dataset  
# This interactive app allows you to explore and compare three different face recognition techniques on low-resolution images.
# """)

# # Load dataset
# data_path = "data/orl_dataset"
# images, labels = model.load_orl_dataset(data_path)
# processed_images = model.preprocess_dataset(images)

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.4, random_state=42)

# # Model selection
# st.markdown("### 🛠️ Choose the Feature Extraction Technique")
# model_choice = st.radio("", ("SIFT", "HOG", "CNN"))

# predictions = []
# accuracy = 0.0

# # Speedometer visualization
# def show_speedometer(accuracy):
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=accuracy * 100,
#         title={'text': "Recognition Accuracy (%)"},
#         gauge={
#             'axis': {'range': [0, 100]},
#             'bar': {'color': "green"},
#             'steps': [
#                 {'range': [0, 50], 'color': "lightcoral"},
#                 {'range': [50, 75], 'color': "khaki"},
#                 {'range': [75, 100], 'color': "lightgreen"}
#             ]
#         }
#     ))
#     st.plotly_chart(fig, use_container_width=True)

# # Handle model logic
# if model_choice == "SIFT":
#     st.markdown("#### 🔑 SIFT Feature Matching in Progress...")
#     train_descriptors = model.extract_sift_features(X_train)
#     test_descriptors = model.extract_sift_features(X_test)
#     predictions = model.match_descriptors(train_descriptors, test_descriptors, y_train)
#     accuracy = accuracy_score(y_test, predictions)
#     show_speedometer(accuracy)

# elif model_choice == "HOG":
#     st.markdown("#### 📐 Extracting HOG Features & Matching...")
#     train_hog_features = model.extract_hog_features(X_train)
#     test_hog_features = model.extract_hog_features(X_test)

#     from sklearn.neighbors import KNeighborsClassifier
#     knn = KNeighborsClassifier(n_neighbors=3)
#     knn.fit(train_hog_features, y_train)
#     predictions = knn.predict(test_hog_features)
#     accuracy = accuracy_score(y_test, predictions)
#     show_speedometer(accuracy)

# elif model_choice == "CNN":
#     st.markdown("#### 🧠 CNN Classification Workflow")
#     if st.button("🚀 Train CNN Model"):
#         model.train_cnn(X_train, y_train, X_test, y_test)
#         st.success("✅ CNN Model Trained Successfully!")

#     if model.cnn_model:
#         predictions = model.predict_cnn(X_test)
#         accuracy = accuracy_score(y_test, predictions)
#         show_speedometer(accuracy)

#         st.markdown("### 📸 Visualizing Individual Predictions")
#         for i in range(0, len(X_test), 3):
#             cols = st.columns(3)
#             for j in range(3):
#                 idx = i + j
#                 if idx < len(X_test):
#                     with cols[j]:
#                         st.image(X_test[idx], caption=f"Pred: {predictions[idx]} | True: {y_test[idx]}", use_container_width=True)
#                         correct = predictions[idx] == y_test[idx]
#                         color = "green" if correct else "red"
#                         label = "✔ Correct" if correct else "✖ Wrong"
#                         st.markdown(f"<div style='text-align:center; color:{color}; font-weight:bold'>{label}</div>", unsafe_allow_html=True)

# # Expandable section for full prediction log
# with st.expander("📋 Show All Predictions"):
#     cols = st.columns(5)
#     for i in range(len(X_test)):
#         with cols[i % 5]:
#             st.image(X_test[i], caption=f"Pred: {predictions[i]} | True: {y_test[i]}", width=120)
