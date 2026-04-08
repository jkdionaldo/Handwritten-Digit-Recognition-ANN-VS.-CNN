from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def preprocess_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = cv2.bitwise_not(image)

    image_vector = image.astype("float32") / 255.0
    return image_vector.flatten()


def preprocess_folder(input_dir: Path) -> list[np.ndarray]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    image_files = sorted(
        [path for path in input_dir.iterdir() if path.suffix.lower() in ALLOWED_EXTENSIONS]
    )

    if not image_files:
        raise ValueError(
            f"No image files found in {input_dir}. Supported extensions: {sorted(ALLOWED_EXTENSIONS)}"
        )

    vectors: list[np.ndarray] = []
    for image_path in image_files:
        vector = preprocess_image(image_path)
        vectors.append(vector)
        print(f"Processed {image_path.name} -> vector length {len(vector)}")

    return vectors


def save_to_csv(vectors: list[np.ndarray], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [f"pixel{i}" for i in range(784)]
    df = pd.DataFrame(vectors, columns=columns)
    df.to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert handwritten digit images into a CSV file for model prediction."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing handwritten digit images.",
    )
    parser.add_argument(
        "--output-csv",
        default="test1.csv",
        help="Path to output CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    vectors = preprocess_folder(input_dir)
    save_to_csv(vectors, output_csv)

    print(f"Saved {len(vectors)} samples to: {output_csv}")


if __name__ == "__main__":
    main()