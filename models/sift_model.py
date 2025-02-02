# models/sift_model.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

class SIFTModel:
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def load_orl_dataset(self, data_path):
        images = []
        labels = []
        for subject in os.listdir(data_path):
            subject_path = os.path.join(data_path, subject)
            for image_name in os.listdir(subject_path):
                image_path = os.path.join(subject_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                images.append(image)
                labels.append(int(subject[1:]))  # Extract subject ID from folder name (e.g., s1, s2)
        return np.array(images), np.array(labels)

    def preprocess_dataset(self, images, target_size=(72, 72)):
        processed_images = []
        for image in images:
            # Resize to target size
            resized_image = cv2.resize(image, target_size)
            # Add Gaussian noise
            noisy_image = resized_image + np.random.normal(0, 10, resized_image.shape).astype(np.uint8)
            processed_images.append(noisy_image)
        return np.array(processed_images)

    def extract_sift_features(self, images):
        descriptors_list = []
        for image in images:
            keypoints, descriptors = self.sift.detectAndCompute(image, None)
            if descriptors is not None:
                descriptors_list.append(descriptors)
        return descriptors_list

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