# Face Detection Model

This repository contains a face detection model built using Python and TensorFlow. The model is designed to detect faces in real-time from a video feed, such as a webcam. The project includes training scripts, the trained model, and a real-time face detection application.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Real-Time Face Detection](#real-time-face-detection)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites
- Python 3.7 or higher
- TensorFlow 2.x
- OpenCV
- NumPy

### Install Required Packages
To install the required packages, run:
```bash
pip install tensorflow opencv-python-headless numpy
# Usage
##  Loading the Model
### First, load your pre-trained model:
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
facetracker = load_model('facetracker.keras')
# Real-Time Face Detection
 ## To use the real-time face detection application, run the following script:
import cv2
import numpy as np
import tensorflow as tf

# Load the facetracker model
facetracker = load_model('facetracker.keras')

# Open video capture (change the index if necessary)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = frame[50:500, 50:500, :]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))
    input_tensor = np.expand_dims(resized / 255.0, axis=0)

    yhat = facetracker.predict(input_tensor)
    sample_coords = yhat[1][0]

    if yhat[0] > 0.5:
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)), 
                      (255, 0, 0), 2)
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [80, 0])), 
                      (255, 0, 0), -1)
        cv2.putText(frame, 'face', 
                    tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('EyeTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Model Training
To train the face detection model, follow these steps:

## Prepare Your Dataset: Ensure you have a labeled dataset of faces.
## Define Your Model: Create a TensorFlow model for face detection.
##  Compile and Train Your Mod
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10, validation_data=val_data)
# Save Your Trained Model
model.save('facetracker.keras')
# Troubleshooting
If you encounter any issues, consider the following tips:

Ensure your camera index is correct.
Verify that OpenCV and TensorFlow are properly installed.
Check your model's predictions to ensure they are as expected.
# Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

# License

This `README.md` provides a comprehensive overview of your face detection project, including installation instructions, usage examples, and troubleshooting tips. Adjust the content as necessary to match the specifics of your project.
![istockphoto-1491338253-1024x1024](https://github.com/Asthagupta622/FaceRecognizitionSystem-using-Tensorflow-and-python/assets/144714106/397bbb0f-f250-4067-8d8f-650eb8f79e44)
