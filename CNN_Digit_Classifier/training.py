import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# Configuration
CONFIG = {
    "path": os.path.join(os.path.dirname(__file__), "db"),
    "test_ratio": 0.2,
    "val_ratio": 0.2,
    "img_dimension": (32, 32, 3),
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
}


def load_data(path):
    """Load and preprocess images from directory structure"""
    images = []  # stores the images
    class_labels = []  # stores the labels

    try:
        class_dirs = [
            d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
        ]
        # Ensuring class directories are sorted numerically if they represent class labels
        class_dirs.sort(key=int)
        # print(f"Total number of classes detected: {len(class_dirs)}")
        num_classes = len(class_dirs)

        for class_idx_str in class_dirs:
            class_idx = int(class_idx_str)
            class_path = os.path.join(path, class_idx_str)
            if not os.path.exists(class_path):
                print(f"Warning: Class directory {class_path} not found")
                continue

            image_files = os.listdir(class_path)
            print(f"Loading class {class_idx}: {len(image_files)} images")

            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)

                if img is not None:
                    # Resize image - preprocess_image will handle grayscale
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
        rotation_range=10,  # Reduced range slightly for digits
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
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


def my_model(num_classes, input_shape):
    """My CNN model for digit recognition"""
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 500
    model = Sequential(
        [
            Conv2D(
                noOfFilters,
                sizeOfFilter1,
                input_shape=input_shape,
                activation="relu",
            ),
            Conv2D(
                noOfFilters,
                sizeOfFilter1,
                activation="relu",
            ),
            MaxPooling2D(pool_size=sizeOfPool),
            Conv2D(
                noOfFilters // 2,
                (sizeOfFilter2[0] // 2, sizeOfFilter2[1] // 2),  # FIXED LINE
                activation="relu",
            ),
            Conv2D(
                noOfFilters,
                sizeOfFilter2,
                activation="relu",
            ),
            MaxPooling2D(pool_size=sizeOfPool),
            Dropout(0.5),
            Flatten(),
            Dense(noOfNode, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

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
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,  # Increased patience
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=10,
            min_lr=1e-7,
            verbose=1,  # Increased patience
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

    if images is None or num_classes == 0:
        print(
            "Failed to load data or no classes found. Please check your data directory and structure."
        )
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
        test_size=CONFIG[
            "val_ratio"
        ],  # This is val_ratio of the *remaining* data (after test split)
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
    # Apply preprocessing to all sets
    X_train = np.array([preprocess_image(img) for img in X_train])
    X_val = np.array([preprocess_image(img) for img in X_val])
    X_test = np.array([preprocess_image(img) for img in X_test])

    # Reshape for CNN (add channel dimension for grayscale)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    # Convert labels to categorical (one-hot encoding)
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Create data generators
    train_generator, val_generator = create_data_generators(
        X_train, y_train, X_val, y_val
    )

    # Create model
    print("Creating model...")
    # Input shape is (height, width, channels). For grayscale, channels=1.
    input_shape = (CONFIG["img_dimension"][0], CONFIG["img_dimension"][1], 1)
    model = my_model(num_classes, input_shape)

    # Print model summary
    model.summary()

    # Create callbacks
    callbacks = create_callbacks()

    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // CONFIG["batch_size"]
    validation_steps = len(X_val) // CONFIG["batch_size"]
    if (
        steps_per_epoch == 0
    ):  # Ensure steps_per_epoch is at least 1 for very small datasets
        steps_per_epoch = 1
    if validation_steps == 0:
        validation_steps = 1

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
