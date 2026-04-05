from pathlib import Path
import tensorflow as tf

# Constants
DATA_DIR = Path("data/raw")
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 123
AUTOTUNE = tf.data.AUTOTUNE


def load_datasets():
    """
    Load training and validation datasets from the image directory.
    Expects:
        data/raw/cats
        data/raw/dogs
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def main():
    train_ds, val_ds, class_names = load_datasets()

    print("Training dataset loaded successfully.")
    print("Validation dataset loaded successfully.")
    print("Class names:", class_names)

    for images, labels in train_ds.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Pixel range:", float(tf.reduce_min(images)), "to", float(tf.reduce_max(images)))


if __name__ == "__main__":
    main()