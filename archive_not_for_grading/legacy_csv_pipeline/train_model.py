from __future__ import annotations
# pyright: reportMissingModuleSource=false

import argparse
import math
import random
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def build_model(input_dim: int = 784, num_classes: int = 10) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_mnist_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train_raw, y_train), (x_test_raw, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train_raw.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test_raw.reshape(-1, 784).astype("float32") / 255.0
    y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=10)
    y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train_one_hot, x_test, y_test_one_hot, x_test_raw


def plot_training_history(history: keras.callbacks.History, output_path: str) -> None:
    history_path = Path(output_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(history_path, dpi=150)
    plt.close(fig)


def show_sample_predictions(model: keras.Model, x_test_flat: np.ndarray, x_test_raw: np.ndarray) -> None:
    sample_count = 16
    predictions = model.predict(x_test_flat[:sample_count], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    num_cells = math.ceil(math.sqrt(sample_count))
    plt.figure(figsize=(6, 6))
    for i in range(sample_count):
        plt.subplot(num_cells, num_cells, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_test_raw[i], cmap="binary")
        plt.xlabel(int(predicted_labels[i]))
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ANN on MNIST handwritten digits.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of training data used for validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training.")
    parser.add_argument("--model-path", default="mnist_model.h5", help="Path to save the trained model.")
    parser.add_argument(
        "--history-plot",
        default="training_history.png",
        help="Path to save the training history plot.",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Show sample predictions after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    x_train, y_train, x_test, y_test, x_test_raw = load_mnist_data()
    model = build_model()

    history = model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Saved model to: {model_path}")

    if args.history_plot:
        plot_training_history(history, args.history_plot)
        print(f"Saved training plot to: {args.history_plot}")

    if args.show_samples:
        show_sample_predictions(model, x_test, x_test_raw)


if __name__ == "__main__":
    main()