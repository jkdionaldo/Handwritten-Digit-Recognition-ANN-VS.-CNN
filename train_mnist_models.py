from __future__ import annotations
# pyright: reportMissingModuleSource=false

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


NUM_CLASSES = 10
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28


def load_and_preprocess_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train_ann = x_train.reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH))
    x_test_ann = x_test.reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH))

    x_train_cnn = np.expand_dims(x_train, axis=-1)
    x_test_cnn = np.expand_dims(x_test, axis=-1)

    y_train_ohe = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_ohe = keras.utils.to_categorical(y_test, NUM_CLASSES)

    return x_train_ann, x_test_ann, x_train_cnn, x_test_cnn, y_train_ohe, y_test_ohe


def build_ann() -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="mnist_ann",
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn() -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="mnist_cnn",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ANN and CNN models on MNIST, evaluate them, and save models for later use."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for both models.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for both models.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Validation split ratio for training data.",
    )
    parser.add_argument(
        "--ann-model-path",
        default="mnist_ann.keras",
        help="Output path for ANN model (.keras or .h5).",
    )
    parser.add_argument(
        "--cnn-model-path",
        default="mnist_cnn.keras",
        help="Output path for CNN model (.keras or .h5).",
    )
    parser.set_defaults(wait_on_exit=True)
    parser.add_argument(
        "--wait-on-exit",
        dest="wait_on_exit",
        action="store_true",
        help="Wait for Enter key before closing so you can review training logs (default).",
    )
    parser.add_argument(
        "--no-wait-on-exit",
        dest="wait_on_exit",
        action="store_false",
        help="Close immediately after training finishes.",
    )
    return parser.parse_args()


def maybe_wait_on_exit(wait_on_exit: bool) -> None:
    if not wait_on_exit:
        return

    prompt = "\nTraining is done. Press Enter to close... "

    try:
        if sys.stdin is not None and sys.stdin.isatty():
            input(prompt)
            return
    except (EOFError, OSError):
        pass

    if os.name == "nt":
        # Fallback for launch modes where stdin is not attached to the console.
        os.system("pause")
        return

    print(prompt, end="")
    try:
        input()
    except EOFError:
        pass


def train_and_evaluate_ann(
    x_train_ann: np.ndarray,
    y_train_ohe: np.ndarray,
    x_val_ann: np.ndarray,
    y_val_ohe: np.ndarray,
    x_test_ann: np.ndarray,
    y_test_ohe: np.ndarray,
    epochs: int,
    batch_size: int,
) -> tuple[keras.Model, float, float]:
    ann_model = build_ann()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
    ]

    print("\nTraining ANN model...")
    ann_model.fit(
        x_train_ann,
        y_train_ohe,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val_ann, y_val_ohe),
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    ann_loss, ann_accuracy = ann_model.evaluate(x_test_ann, y_test_ohe, verbose=0)
    print(f"ANN Test Loss: {ann_loss:.4f}")
    print(f"ANN Test Accuracy: {ann_accuracy:.4f}")
    return ann_model, ann_loss, ann_accuracy


def train_and_evaluate_cnn(
    x_train_cnn: np.ndarray,
    y_train_ohe: np.ndarray,
    x_val_cnn: np.ndarray,
    y_val_ohe: np.ndarray,
    x_test_cnn: np.ndarray,
    y_test_ohe: np.ndarray,
    epochs: int,
    batch_size: int,
) -> tuple[keras.Model, float, float]:
    cnn_model = build_cnn()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
    ]

    print("\nTraining CNN model...")
    cnn_model.fit(
        x_train_cnn,
        y_train_ohe,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val_cnn, y_val_ohe),
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    cnn_loss, cnn_accuracy = cnn_model.evaluate(x_test_cnn, y_test_ohe, verbose=0)
    print(f"CNN Test Loss: {cnn_loss:.4f}")
    print(f"CNN Test Accuracy: {cnn_accuracy:.4f}")
    return cnn_model, cnn_loss, cnn_accuracy


def ensure_parent_dir(file_path: str) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)


def split_indices_by_class(
    y_one_hot: np.ndarray,
    validation_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1.")

    labels = np.argmax(y_one_hot, axis=1)
    rng = np.random.default_rng(seed)

    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []
    for digit in range(NUM_CLASSES):
        indices = np.where(labels == digit)[0]
        rng.shuffle(indices)

        val_count = max(1, int(len(indices) * validation_split))
        val_indices.append(indices[:val_count])
        train_indices.append(indices[val_count:])

    train_idx = np.concatenate(train_indices)
    val_idx = np.concatenate(val_indices)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def report_per_digit_accuracy(model: keras.Model, x_test: np.ndarray, y_test_ohe: np.ndarray, model_name: str) -> None:
    y_true = np.argmax(y_test_ohe, axis=1)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    print(f"\n{model_name} per-digit accuracy:")
    for digit in range(NUM_CLASSES):
        mask = y_true == digit
        digit_accuracy = float(np.mean(y_pred[mask] == y_true[mask]))
        print(f"Digit {digit}: {digit_accuracy:.4f} ({int(np.sum(mask))} samples)")


def main() -> None:
    args = parse_args()
    try:
        keras.utils.set_random_seed(args.seed)

        x_train_ann, x_test_ann, x_train_cnn, x_test_cnn, y_train_ohe, y_test_ohe = (
            load_and_preprocess_data()
        )

        train_idx, val_idx = split_indices_by_class(
            y_one_hot=y_train_ohe,
            validation_split=args.validation_split,
            seed=args.seed,
        )

        x_ann_train, y_ann_train = x_train_ann[train_idx], y_train_ohe[train_idx]
        x_ann_val, y_ann_val = x_train_ann[val_idx], y_train_ohe[val_idx]

        x_cnn_train, y_cnn_train = x_train_cnn[train_idx], y_train_ohe[train_idx]
        x_cnn_val, y_cnn_val = x_train_cnn[val_idx], y_train_ohe[val_idx]

        ann_model, _, _ = train_and_evaluate_ann(
            x_train_ann=x_ann_train,
            y_train_ohe=y_ann_train,
            x_val_ann=x_ann_val,
            y_val_ohe=y_ann_val,
            x_test_ann=x_test_ann,
            y_test_ohe=y_test_ohe,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        cnn_model, _, _ = train_and_evaluate_cnn(
            x_train_cnn=x_cnn_train,
            y_train_ohe=y_cnn_train,
            x_val_cnn=x_cnn_val,
            y_val_ohe=y_cnn_val,
            x_test_cnn=x_test_cnn,
            y_test_ohe=y_test_ohe,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        report_per_digit_accuracy(ann_model, x_test_ann, y_test_ohe, model_name="ANN")
        report_per_digit_accuracy(cnn_model, x_test_cnn, y_test_ohe, model_name="CNN")

        ensure_parent_dir(args.ann_model_path)
        ensure_parent_dir(args.cnn_model_path)

        ann_model.save(args.ann_model_path)
        cnn_model.save(args.cnn_model_path)

        print("\nSaved ANN model to:", args.ann_model_path)
        print("Saved CNN model to:", args.cnn_model_path)
    finally:
        maybe_wait_on_exit(args.wait_on_exit)


if __name__ == "__main__":
    main()
