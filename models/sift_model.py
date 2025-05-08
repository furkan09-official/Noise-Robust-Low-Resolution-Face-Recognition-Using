
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from skimage.feature import hog
from scipy.spatial.distance import cdist  # For SIFT descriptor matching
from sklearn.model_selection import train_test_split

class FaceRecognitionModel:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.cnn_model = None  # Will initialize in train_cnn()

    def load_orl_dataset(self, data_path):
        images = []
        labels = []
        for subject in os.listdir(data_path):
            subject_path = os.path.join(data_path, subject)
            if not os.path.isdir(subject_path):
                continue
            for image_name in os.listdir(subject_path):
                image_path = os.path.join(subject_path, image_name)
                if not os.path.isfile(image_path):
                    continue
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    images.append(image)
                    labels.append(int(subject[1:]))  # Extract subject ID from folder name (e.g., s1 → 1)
        return np.array(images), np.array(labels)

    def preprocess_dataset(self, images, target_size=(72, 72)):
        processed_images = []
        for image in images:
            resized_image = cv2.resize(image, target_size)
            noisy_image = resized_image + np.random.normal(0, 10, resized_image.shape).astype(np.int16)
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            processed_images.append(noisy_image)
        return np.array(processed_images)

    def extract_sift_features(self, images):
        descriptors_list = []
        for image in images:
            keypoints, descriptors = self.sift.detectAndCompute(image, None)
            if descriptors is not None:
                descriptors_list.append(descriptors)
        return descriptors_list

    def extract_hog_features(self, images):
        hog_features = []
        for image in images:
            features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
            hog_features.append(features)
        return np.array(hog_features)

    def match_descriptors(self, train_descriptors, test_descriptors, y_train):
        predictions = []
        for test_desc in test_descriptors:
            distances = []
            for train_desc in train_descriptors:
                if train_desc is not None and test_desc is not None:
                    dist = cdist(test_desc, train_desc, metric='euclidean')
                    min_dist = np.min(dist, axis=1)
                    distances.append(np.mean(min_dist))
                else:
                    distances.append(np.inf)
            predicted_label = y_train[np.argmin(distances)]
            predictions.append(predicted_label)
        return np.array(predictions)

    def build_cnn(self, input_shape):
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(42, activation="softmax")  # 40 subjects in ORL dataset
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def train_cnn(self, X_train, y_train, X_val, y_val):
        X_train = X_train.reshape((-1, 72, 72, 1)) / 255.0
        X_val = X_val.reshape((-1, 72, 72, 1)) / 255.0
        self.cnn_model = self.build_cnn((72, 72, 1))
        self.cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

    def predict_cnn(self, X_test):
        if self.cnn_model is None:
            raise ValueError('the cnn model has not been trained yet')
        X_test = X_test.reshape((-1, 72, 72, 1)) / 255.0
        predictions = np.argmax(self.cnn_model.predict(X_test), axis=1)
        return predictions









