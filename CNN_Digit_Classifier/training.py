import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# Configuration
CONFIG = {
    "path": "db",
    "test_ratio": 0.2,
    "val_ratio": 0.2,
    "img_dimension": (64, 64, 3),  # Increased size for better feature extraction
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
}


def load_data(path):
    """Load and preprocess images from directory structure"""
    images = []
    class_labels = []

    try:
        class_dirs = os.listdir(path)
        print(f"Total number of classes detected: {len(class_dirs)}")
        num_classes = len(class_dirs)

        for class_idx in range(num_classes):
            class_path = os.path.join(path, str(class_idx))
            if not os.path.exists(class_path):
                print(f"Warning: Class directory {class_path} not found")
                continue

            image_files = os.listdir(class_path)
            print(f"Loading class {class_idx}: {len(image_files)} images")

            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)

                if img is not None:
                    # Resize image
                    img = cv2.resize(
                        img, (CONFIG["img_dimension"][0], CONFIG["img_dimension"][1])
                    )
                    images.append(img)
                    class_labels.append(class_idx)
                else:
                    print(f"Warning: Could not read image {img_path}")

        return np.array(images), np.array(class_labels), num_classes

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, 0


def preprocess_image(img):
    """Enhanced preprocessing with normalization"""
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization for better contrast
    img = cv2.equalizeHist(img)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    return img


def create_data_generators(X_train, y_train, X_val, y_val):
    """Create data generators with augmentation"""
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Validation data generator (no augmentation)
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        X_train, y_train, batch_size=CONFIG["batch_size"], shuffle=True
    )

    val_generator = val_datagen.flow(
        X_val, y_val, batch_size=CONFIG["batch_size"], shuffle=False
    )

    return train_generator, val_generator


def create_improved_model(num_classes, input_shape):
    """Create an improved CNN model with modern architecture"""
    model = Sequential(
        [
            # First Convolutional Block
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation="relu"),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation="relu"),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation="relu"),
            BatchNormalization(),
            Dropout(0.25),

            GlobalAveragePooling2D(),
            # Dense layers
            Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            # Output layer
            Dense(num_classes, activation="softmax"),
        ]
    )

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=CONFIG["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy", "top_k_categorical_accuracy"],
    )

    return model


def create_callbacks():
    """Create training callbacks"""
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=8, min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            "best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]
    return callbacks


def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot training & validation accuracy
    ax1.plot(history.history["accuracy"], label="Training Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Plot training & validation loss
    ax2.plot(history.history["loss"], label="Training Loss")
    ax2.plot(history.history["val_loss"], label="Validation Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """Main training pipeline"""
    print("Loading data...")
    images, class_labels, num_classes = load_data(CONFIG["path"])

    if images is None:
        print("Failed to load data. Please check your data directory.")
        return

    print(f"Total images loaded: {len(images)}")
    print(f"Number of classes: {num_classes}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images,
        class_labels,
        test_size=CONFIG["test_ratio"],
        random_state=42,
        stratify=class_labels,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=CONFIG["val_ratio"],
        random_state=42,
        stratify=y_train,
    )

    print(f"Training set: {len(X_train)}")
    print(f"Validation set: {len(X_val)}")
    print(f"Test set: {len(X_test)}")

    # Plot class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    plt.figure(figsize=(12, 6))
    plt.bar(unique, counts)
    plt.title("Training Set Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.xticks(unique)
    plt.grid(True, alpha=0.3)
    plt.show()

    # Preprocess images
    print("Preprocessing images...")
    X_train = np.array([preprocess_image(img) for img in X_train])
    X_val = np.array([preprocess_image(img) for img in X_val])
    X_test = np.array([preprocess_image(img) for img in X_test])

    # Reshape for CNN (add channel dimension)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    # Convert labels to categorical
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Create data generators
    train_generator, val_generator = create_data_generators(
        X_train, y_train, X_val, y_val
    )

    # Create model
    print("Creating model...")
    input_shape = (CONFIG["img_dimension"][0], CONFIG["img_dimension"][1], 1)
    model = create_improved_model(num_classes, input_shape)

    # Print model summary
    model.summary()

    # Create callbacks
    callbacks = create_callbacks()

    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // CONFIG["batch_size"]
    validation_steps = len(X_val) // CONFIG["batch_size"]

    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=CONFIG["epochs"],
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_accuracy, test_top_k = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Top-K Accuracy: {test_top_k:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Save final model
    model.save("final_model.h5")
    print("Model saved as 'final_model.h5'")

    return model, history


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Run main pipeline
    model, history = main()
