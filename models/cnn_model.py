# models/cnn_model.py
# import tensorflow as tf
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input

# class CNNModel:
#     def __init__(self):
#         # Load pre-trained VGG16 model (without the top classification layer)
#         self.model = VGG16(weights='imagenet', include_top=False, input_shape=(72, 72, 3))

#     def extract_cnn_features(self, images):
#         cnn_features = []
#         for image in images:
#             # Convert grayscale image to RGB (required for VGG16)
#             if len(image.shape) == 2:
#                 image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#             # Resize image to match VGG16 input size
#             resized_image = cv2.resize(image, (72, 72))
#             # Preprocess image for VGG16
#             preprocessed_image = preprocess_input(resized_image)
#             # Expand dimensions to match VGG16 input shape
#             preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
#             # Extract features using VGG16
#             features = self.model.predict(preprocessed_image)
#             cnn_features.append(features.flatten())
#         return np.array(cnn_features)