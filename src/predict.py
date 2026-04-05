from pathlib import Path
import sys
import tensorflow as tf
from tensorflow import keras

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pet_classifier.keras"
IMAGE_SIZE = (180, 180)


def predict_image(image_path):
    model = keras.models.load_model(MODEL_PATH)

    img = keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
    img_array = img_array / 255.0

    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction >= 0.5:
        predicted_class = "dogs"
        confidence = prediction
    else:
        predicted_class = "cats"
        confidence = 1 - prediction

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Error: file not found -> {image_path}")
        return

    predict_image(image_path)


if __name__ == "__main__":
    main()