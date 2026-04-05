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
│   ├── validate_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt

Tech Stack
Python 3.11
TensorFlow / Keras
Pandas
Matplotlib
Pillow
Git / GitHub


## How to Run

# validate dataset
cd src
python validate_data.py

# Activate environment
venv\Scripts\activate

# Train model
cd src
python train_model.py

# Evaluate model
python evaluate_model.py

# Predict image
python predict.py ..\data\raw\dogs\images.jpg

```md
## Notes
This project demonstrates a modular machine learning pipeline with data validation, preprocessing, training, evaluation, visualization, and prediction logging. Model performance is limited by dataset size, which highlights the importance of data quality and scale in real-world workflows.