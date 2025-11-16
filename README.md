# CROP-Disease-Classification
🌱 Potato Disease Classification using CNN (TensorFlow)

<img src="https://github.com/Subhashdrx2002/Potato-Disease-Classification/blob/main/assets/potato_sample.jpg" alt="Potato Disease Sample" width="600">


A single-file, end-to-end pipeline that trains a Convolutional Neural Network to classify potato leaf images into **Early Blight**, **Late Blight**, and **Healthy** using the PlantVillage dataset. Includes dataset loading, augmentation, training, evaluation, inference and model saving.


"""
Crop_classification_full.py
Full end-to-end single-file script for Potato Disease Classification
Requirements:
    pip install tensorflow matplotlib numpy
Place the 'PlantVillage' folder in the same directory as this script.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers

# ----------------------
# Constants / Hyperparams
# ----------------------
BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS = 3
EPOCHS = 50
SEED = 123
DATA_DIR = "PlantVillage"   # change if your dataset folder name differs
AUTOTUNE = tf.data.AUTOTUNE

# ----------------------
# Helper: print environment info (optional)
# ----------------------
def print_env():
    print("TensorFlow version:", tf.__version__)
    print("Data directory:", DATA_DIR)
    print("Batch size:", BATCH_SIZE, "Image size:", IMAGE_SIZE, "Epochs:", EPOCHS)

# ----------------------
# Load dataset with tf.keras.preprocessing.image_dataset_from_directory
# ----------------------
def load_image_dataset(data_dir=DATA_DIR):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        seed=SEED,
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )
    class_names = ds.class_names
    print("Found", ds.cardinality().numpy() * 1, "batches (batch size {})".format(BATCH_SIZE))
    print("Classes:", class_names)
    return ds, class_names

# ----------------------
# Partition dataset into train/val/test
# ----------------------
def get_dataset_partitions_tf(dataset, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    """
    Partition a tf.data.Dataset of batches into train/validation/test datasets.
    This preserves batch structure.
    """
    assert (train_split + val_split + test_split) == 1.0

    ds_size = tf.data.experimental.cardinality(dataset).numpy()
    if ds_size == tf.data.experimental.UNKNOWN_CARDINALITY:
        raise ValueError("Dataset cardinality unknown. Make sure dataset is finite (image_dataset_from_directory returns finite dataset).")

    if shuffle:
        dataset = dataset.shuffle(shuffle_size, seed=SEED)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

# ----------------------
# Data augmentation & preprocessing layers
# ----------------------
def get_preprocessing_and_augmentation():
    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.experimental.preprocessing.Rescaling(1.0 / 255.0),
    ], name="resize_and_rescale")

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ], name="data_augmentation")

    return resize_and_rescale, data_augmentation

# ----------------------
# Build CNN model
# ----------------------
def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), n_classes=3):
    resize_and_rescale, _ = get_preprocessing_and_augmentation()

    model = models.Sequential([
        resize_and_rescale,
        layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ], name="potato_cnn")

    return model

# ----------------------
# Plot training history
# ----------------------
def plot_history(history):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.show()

# ----------------------
# Inference helper
# ----------------------
def predict_image(model, img_array, class_names):
    """
    img_array: either a numpy array shape (H,W,3) with pixel values [0,255] or already preprocessed image
    returns (predicted_class_name, confidence_percent)
    """
    # If values are ints 0-255, convert to float and rescale using model preprocessing
    if img_array.dtype != np.float32 and img_array.max() > 1.0:
        img_array = img_array.astype("float32")

    img = np.expand_dims(img_array, axis=0)   # shape (1, H, W, 3)
    preds = model.predict(img)
    idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]) * 100.0)
    return class_names[idx], round(confidence, 2)

# ----------------------
# Main pipeline
# ----------------------
def main():
    print_env()

    # Load dataset
    dataset, class_names = load_image_dataset(DATA_DIR)
    n_classes = len(class_names)

    # Partition
    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True)

    # Get augmentation and preprocessors
    resize_and_rescale, data_augmentation = get_preprocessing_and_augmentation()

    # Apply augmentation to train dataset only (we keep batches consistent)
    # Each element in the datasets is (batch_images, batch_labels) because we loaded with batch_size
    def apply_augmentation(images, labels):
        images = data_augmentation(images, training=True)
        return images, labels

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y)).cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (resize_and_rescale(x), y)).cache().prefetch(AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (resize_and_rescale(x), y)).cache().prefetch(AUTOTUNE)

    # Build & compile model
    model = build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), n_classes=n_classes)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    model.summary()

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )

    # Plot training history
    plot_history(history)

    # Evaluate on test set
    print("\nEvaluating on test dataset...")
    scores = model.evaluate(test_ds)
    print("Test loss, Test accuracy:", scores)

    # Run prediction on a few test images (visual)
    print("\nRunning inference on a few test images (showing up to 9)...")
    for images_batch, labels_batch in test_ds.take(1):
        # images_batch is a batch (BATCH_SIZE, H, W, 3)
        to_show = min(9, images_batch.shape[0])
        plt.figure(figsize=(9, 9))
        for i in range(to_show):
            img = images_batch[i].numpy()
            true_label = class_names[int(labels_batch[i].numpy())]
            pred_label, conf = predict_image(model, img, class_names)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow((img * 255).astype("uint8"))  # images are rescaled (0..1), so multiply back for display
            plt.title(f"Actual: {true_label}\nPred: {pred_label} ({conf}%)")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    # Save model: simple .h5 and versioned folder
    save_name = "potatoes.h5"
    print(f"Saving model to {save_name} ...")
    model.save(save_name)
    print("Model saved.")

    # Optionally, demonstrate loading model (uncomment to test)
    # loaded = tf.keras.models.load_model(save_name)
    # print("Loaded model summary:")
    # loaded.summary()

if __name__ == "__main__":
    main()
