from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def load_input_csv(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    data = pd.read_csv(csv_path)
    if data.empty:
        raise ValueError("Input CSV is empty.")

    if data.shape[1] != 784:
        raise ValueError(f"Expected 784 columns, but got {data.shape[1]} columns.")

    x_input = data.to_numpy(dtype="float32")

    if x_input.max() > 1.0:
        x_input = x_input / 255.0

    return x_input


def show_predictions(x_input: np.ndarray, labels: np.ndarray, confidences: np.ndarray) -> None:
    count = min(len(labels), 16)
    num_cells = math.ceil(math.sqrt(count))

    plt.figure(figsize=(7, 7))
    for i in range(count):
        plt.subplot(num_cells, num_cells, i + 1)
        plt.xticks([])
        plt.yticks([])
        image = x_input[i].reshape((28, 28))
        plt.imshow(image, cmap="binary")
        plt.xlabel(f"{labels[i]} ({confidences[i] * 100:.1f}%)")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict handwritten digits from CSV using a trained ANN.")
    parser.add_argument("--model-path", default="mnist_model.h5", help="Path to the trained model file.")
    parser.add_argument("--input-csv", default="test1.csv", help="Path to handwritten digits CSV.")
    parser.add_argument(
        "--output-csv",
        default="predictions.csv",
        help="Path to save predictions and confidence scores.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display prediction previews in a matplotlib window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path)
    x_input = load_input_csv(input_csv)

    probabilities = model.predict(x_input, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    results = pd.DataFrame(
        {
            "sample_index": np.arange(len(predicted_labels)),
            "predicted_label": predicted_labels,
            "confidence": confidences,
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    print(results.to_string(index=False))
    print(f"Saved predictions to: {output_csv}")

    if args.show:
        show_predictions(x_input, predicted_labels, confidences)


if __name__ == "__main__":
    main()