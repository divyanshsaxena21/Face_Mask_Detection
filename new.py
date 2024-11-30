# Example for testing model prediction
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('face_mask_detection.h5')

# Read and process an image
img = cv2.imread('test_image.jpg')
img = cv2.resize(img, (256, 256))  # Resize to the expected input size
img = img / 255.0  # Normalize
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Make prediction
prediction = model.predict(img)
print(prediction)
