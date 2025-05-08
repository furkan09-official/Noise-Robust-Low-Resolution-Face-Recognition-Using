# # models/hog_model.py
# import cv2
# import numpy as np

# class HOGModel:
#     def __init__(self):
#         self.hog = cv2.HOGDescriptor()

#     def extract_hog_features(self, images):
#         hog_features = []
#         for image in images:
#             # Resize image to a fixed size (required for HOG)
#             resized_image = cv2.resize(image, (64, 128))
#             # Compute HOG features
#             features = self.hog.compute(resized_image)
#             if features is not None:
#                 hog_features.append(features.flatten())
#         return np.array(hog_features)