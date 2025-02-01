import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from feature_extraction import extract_features

# Dataset path
DATASET_PATH = "dataset"

# Prepare dataset
X, y = [], []
for label, subject in enumerate(os.listdir(DATASET_PATH)):  # Loop through subjects
    subject_path = os.path.join(DATASET_PATH, subject)
    if os.path.isdir(subject_path):
        for image in os.listdir(subject_path):
            image_path = os.path.join(subject_path, image)
            X.append(extract_features(image_path))
            y.append(label)

# Convert to NumPy arrays
X, y = np.array(X), np.array(y)

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifiers
models = {
    "SVM": SVC(kernel="linear"),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    joblib.dump(model, f"../models/{name}.pkl")  # Save trained model
