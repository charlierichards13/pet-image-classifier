from pathlib import Path
import tensorflow as tf
from load_data import load_datasets

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pet_classifier.keras"


def main():
    # Load datasets
    train_ds, val_ds, class_names = load_datasets()

    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    
    loss, accuracy = model.evaluate(val_ds)

    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()