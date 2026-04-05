from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from load_data import load_datasets

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pet_classifier.keras"
PLOT_PATH = BASE_DIR / "models" / "training_history.png"
EPOCHS = 10


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


def plot_training_history(history):
    """
    Plot training and validation accuracy/loss over epochs.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Training plot saved to {PLOT_PATH}")


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

    plot_training_history(history)

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()