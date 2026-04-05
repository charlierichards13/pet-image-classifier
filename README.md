# Pet Image Classifier

A Python-based image classification pipeline using TensorFlow and Keras to load, preprocess, train, evaluate, and predict pet images.

## Features
- Loads labeled image data from folder-based classes
- Preprocesses and normalizes image datasets
- Trains a convolutional neural network (CNN)
- Evaluates model performance on validation data
- Predicts whether a new image is a cat or dog
- Saves the trained model for reuse

## Project Structure
```text
pet-image-classifier/
├── data/
│   ├── raw/
│   │   ├── cats/
│   │   └── dogs/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── load_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt