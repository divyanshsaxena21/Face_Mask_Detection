from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('face_mask_detection.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    try:
        # Read the image from the file and decode it
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Resize the image to the required input shape (128x128) instead of 256x256
        img = cv2.resize(img, (128, 128))  # Resize to 128x128 or the size your model expects
        
        # Normalize the image data to [0, 1]
        img = img / 255.0  # Normalize to [0, 1]
        
        # Ensure the image is in the correct dtype (float32)
        img = img.astype(np.float32)
        
        # Add the batch dimension to the image (shape becomes (1, 128, 128, 3))
        img = np.expand_dims(img, axis=0)
        
        # Perform prediction using the model
        prediction = model.predict(img)
        
        # If the model has a single output node, binary classification
        if prediction.shape[1] == 1:
            label = 'Mask' if prediction[0][0] > 0.5 else 'No Mask'
        else:
            # If the model is multi-class, interpret the output
            label = np.argmax(prediction, axis=1)
            class_labels = ['Mask', 'No Mask', 'Incorrect Mask']  # Example class labels, adjust as needed
            label = class_labels[label[0]]

        return jsonify({'prediction': label})
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
