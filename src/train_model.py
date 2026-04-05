# from pathlib import Path
# import tensorflow as tf
# from load_data import load_datasets

# BASE_DIR = Path(__file__).resolve().parent.parent
# MODEL_PATH = BASE_DIR / "models" / "pet_classifier.keras"
# EPOCHS = 10


# def build_model():
#     """
#     Build a simple CNN model for binary image classification.
#     """
#     model = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(16, 3, activation="relu", input_shape=(180, 180, 3)),
#         tf.keras.layers.MaxPooling2D(),

#         tf.keras.layers.Conv2D(32, 3, activation="relu"),
#         tf.keras.layers.MaxPooling2D(),

#         tf.keras.layers.Conv2D(64, 3, activation="relu"),
#         tf.keras.layers.MaxPooling2D(),

#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128, activation="relu"),
#         tf.keras.layers.Dense(1, activation="sigmoid")
#     ])

#     return model


# def main():
#     train_ds, val_ds, class_names = load_datasets()

#     print("Classes:", class_names)

#     model = build_model()

#     model.compile(
#         optimizer="adam",
#         loss="binary_crossentropy",
#         metrics=["accuracy"]
#     )

#     model.summary()

#     history = model.fit(
#         train_ds,
#         validation_data=val_ds,
#         epochs=EPOCHS
#     )

#     model.save(MODEL_PATH)
#     print(f"Model saved to {MODEL_PATH}")


# if __name__ == "__main__":
#     main()




## strenghten data augmentation and add it to the model

from pathlib import Path
import tensorflow as tf
from load_data import load_datasets

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pet_classifier.keras"
EPOCHS = 10
IMAGE_SIZE = (180, 180)


def build_model():
    """
    Build a CNN model for binary image classification
    with data augmentation.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(180, 180, 3)),

        data_augmentation,

        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    return model


def main():
    train_ds, val_ds, class_names = load_datasets()

    print("Classes:", class_names)

    model = build_model()

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()