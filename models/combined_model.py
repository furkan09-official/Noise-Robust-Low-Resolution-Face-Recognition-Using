# # models/combined_model.py
# import numpy as np
# from scipy.spatial.distance import cdist
# from .sift_model import SIFTModel
# from .hog_model import HOGModel
# from .cnn_model import CNNModel

# class CombinedModel:
#     def __init__(self):
#         self.sift_model = SIFTModel()  # SIFT feature extraction
#         self.hog_model = HOGModel()    # HOG feature extraction
#         self.cnn_model = CNNModel()    # CNN feature extraction

#     def extract_features(self, images):
#         """
#         Extract features using SIFT, HOG, and CNN.
#         """
#         # Extract SIFT features
#         sift_features = self.sift_model.extract_sift_features(images)
        
#         # Extract HOG features
#         hog_features = self.hog_model.extract_hog_features(images)
        
#         # Extract CNN features
#         cnn_features = self.cnn_model.extract_cnn_features(images)

#         # Combine features
#         combined_features = []
#         for i in range(len(images)):
#             # Handle cases where SIFT descriptors are not found
#             sift_feat = sift_features[i] if i < len(sift_features) and sift_features[i] is not None else np.array([])
#             # HOG and CNN features are always available
#             hog_feat = hog_features[i] if i < len(hog_features) else np.array([])
#             cnn_feat = cnn_features[i] if i < len(cnn_features) else np.array([])
#             # Combine all features into a single feature vector
#             combined_feat = np.hstack([sift_feat, hog_feat, cnn_feat])
#             combined_features.append(combined_feat)
#         return np.array(combined_features)

#     def match_descriptors(self, train_features, test_features, y_train):
#         """
#         Match descriptors using Euclidean distance.
#         """
#         predictions = []
#         for test_feat in test_features:
#             distances = []
#             for train_feat in train_features:
#                 if train_feat.size > 0 and test_feat.size > 0:
#                     # Compute Euclidean distance between feature vectors
#                     dist = cdist(test_feat.reshape(1, -1), train_feat.reshape(1, -1), metric='euclidean')
#                     distances.append(dist[0][0])
#                 else:
#                     distances.append(np.inf)  # Assign infinite distance if features are empty
#             # Predict the label with the smallest distance
#             predicted_label = y_train[np.argmin(distances)]
#             predictions.append(predicted_label)
#         return np.array(predictions)

# import cv2
# import numpy as np
# from skimage.feature import hog
# from tensorflow.keras.models import load_model
# from scipy.spatial.distance import cdist

# class FeatureExtractor:
#     def __init__(self):
#         self.sift = cv2.SIFT_create()
#         self.cnn_model = load_model("cnn_model.h5")  # Load pre-trained CNN model

#     def extract_sift_features(self, images):
#         descriptors_list = []
#         for image in images:
#             keypoints, descriptors = self.sift.detectAndCompute(image, None)
#             if descriptors is not None:
#                 descriptors_list.append(descriptors)
#         return descriptors_list

#     def extract_hog_features(self, images, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):  
#         hog_features = []
#         for image in images:
#             feature = hog(image, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, feature_vector=True)
#             hog_features.append(feature)
#         return np.array(hog_features)

#     def extract_cnn_features(self, images):
#         images_resized = np.array([cv2.resize(img, (64, 64)) for img in images]) / 255.0  # Normalize
#         features = self.cnn_model.predict(images_resized)
#         return features

#     def match_descriptors(self, train_features, test_features, y_train):
#         predictions = []
#         for test_feature in test_features:
#             distances = cdist([test_feature], train_features, metric='euclidean')
#             predicted_label = y_train[np.argmin(distances)]
#             predictions.append(predicted_label)
#         return np.array(predictions)
