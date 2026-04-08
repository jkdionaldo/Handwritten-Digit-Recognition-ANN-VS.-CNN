from __future__ import annotations
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model


CANVAS_SIZE = 280
IMAGE_SIZE = 28
MNIST_DRAW_SIZE = 20


class DigitRecognizerApp:
    def __init__(self, root: tk.Tk, ann_model_path: str, cnn_model_path: str) -> None:
        self.root = root
        self.root.title("Real-Time Digit")
        self.root.resizable(False, False)
        self.root.configure(bg='#f0f0f0')

        if not Path(ann_model_path).exists():
            raise FileNotFoundError(f"ANN model file not found: {ann_model_path}")
        if not Path(cnn_model_path).exists():
            raise FileNotFoundError(f"CNN model file not found: {cnn_model_path}")

        self.ann_model = load_model(ann_model_path)
        self.cnn_model = load_model(cnn_model_path)

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="gray",
            cursor="cross",
            highlightthickness=1,
            highlightbackground="#626262",
        )
        self.canvas.pack(padx=10, pady=10)

        self.image_array = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        self.last_x: int | None = None
        self.last_y: int | None = None
        self.brush_size = 18

        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        controls_frame = tk.Frame(self.root)
        controls_frame.pack(pady=5)

        predict_button = tk.Button(controls_frame, text="Predict", command=self.predict_digit, width=10)
        predict_button.grid(row=0, column=0, padx=5)

        clear_button = tk.Button(controls_frame, text="Clear", command=self.clear_canvas, width=10)
        clear_button.grid(row=0, column=1, padx=5)

        results_frame = tk.Frame(self.root)
        results_frame.pack(pady=10)

        self.ann_digit_var = tk.StringVar(value="-")
        self.ann_conf_var = tk.StringVar(value="--.-%")
        self.cnn_digit_var = tk.StringVar(value="-")
        self.cnn_conf_var = tk.StringVar(value="--.-%")

        ann_panel = tk.Frame(results_frame, bd=1, relief=tk.SOLID, padx=10, pady=6)
        ann_panel.grid(row=0, column=0, padx=6)
        tk.Label(ann_panel, text="ANN", font=("Arial", 10, "bold"), fg="#1b5e20").pack()
        tk.Label(ann_panel, textvariable=self.ann_digit_var, font=("Arial", 20, "bold"), width=2).pack()
        tk.Label(ann_panel, textvariable=self.ann_conf_var, font=("Arial", 10), width=8).pack()

        cnn_panel = tk.Frame(results_frame, bd=1, relief=tk.SOLID, padx=10, pady=6)
        cnn_panel.grid(row=0, column=1, padx=6)
        tk.Label(cnn_panel, text="CNN", font=("Arial", 10, "bold"), fg="#0d47a1").pack()
        tk.Label(cnn_panel, textvariable=self.cnn_digit_var, font=("Arial", 20, "bold"), width=2).pack()
        tk.Label(cnn_panel, textvariable=self.cnn_conf_var, font=("Arial", 10), width=8).pack()

        self.verdict_label = tk.Label(
            self.root,
            text="Verdict: -",
            font=("Arial", 11, "bold"),
            width=32,
            anchor="center",
        )
        self.verdict_label.pack(pady=2)

    def start_stroke(self, event: tk.Event) -> None:
        self.last_x, self.last_y = event.x, event.y
        self.draw(event)

    def draw(self, event: tk.Event) -> None:
        x, y = event.x, event.y
        radius = self.brush_size // 2

        if self.last_x is None or self.last_y is None:
            self.last_x, self.last_y = x, y

        self.canvas.create_line(
            self.last_x,
            self.last_y,
            x,
            y,
            fill="black",
            width=self.brush_size,
            capstyle=tk.ROUND,
            smooth=True,
        )

        cv2.line(
            self.image_array,
            (self.last_x, self.last_y),
            (x, y),
            color=255,
            thickness=self.brush_size,
            lineType=cv2.LINE_AA,
        )

        cv2.circle(self.image_array, (x, y), radius, color=255, thickness=-1)
        self.last_x, self.last_y = x, y

    def end_stroke(self, _event: tk.Event) -> None:
        self.last_x, self.last_y = None, None

    def _center_and_resize(self, image: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
        points = cv2.findNonZero(binary)

        if points is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        x, y, w, h = cv2.boundingRect(points)
        margin = 20
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(CANVAS_SIZE, x + w + margin), min(CANVAS_SIZE, y + h + margin)
        cropped = binary[y0:y1, x0:x1]

        crop_h, crop_w = cropped.shape
        if crop_h > crop_w:
            new_h = MNIST_DRAW_SIZE
            new_w = max(1, int(round(crop_w * MNIST_DRAW_SIZE / crop_h)))
        else:
            new_w = MNIST_DRAW_SIZE
            new_h = max(1, int(round(crop_h * MNIST_DRAW_SIZE / crop_w)))

        interpolation = cv2.INTER_AREA if max(crop_h, crop_w) > MNIST_DRAW_SIZE else cv2.INTER_CUBIC
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=interpolation)

        mnist_like = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        x_offset = (IMAGE_SIZE - new_w) // 2
        y_offset = (IMAGE_SIZE - new_h) // 2
        mnist_like[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        moments = cv2.moments(mnist_like)
        if moments["m00"] > 0:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
            shift_x = int(round((IMAGE_SIZE / 2) - center_x))
            shift_y = int(round((IMAGE_SIZE / 2) - center_y))
            translation = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            mnist_like = cv2.warpAffine(mnist_like, translation, (IMAGE_SIZE, IMAGE_SIZE), borderValue=0)

        return cv2.GaussianBlur(mnist_like, (3, 3), 0)

    def preprocess_for_mnist(self) -> np.ndarray:
        mnist_like = self._center_and_resize(self.image_array)
        normalized = mnist_like.astype("float32") / 255.0
        return normalized

    def _prepare_ann_input(self, normalized_image: np.ndarray) -> np.ndarray:
        input_shape = self.ann_model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if len(input_shape) == 2 and input_shape[1] == IMAGE_SIZE * IMAGE_SIZE:
            return normalized_image.reshape(1, IMAGE_SIZE * IMAGE_SIZE)
        if len(input_shape) == 3 and input_shape[1] == IMAGE_SIZE and input_shape[2] == IMAGE_SIZE:
            return normalized_image.reshape(1, IMAGE_SIZE, IMAGE_SIZE)
        if (
            len(input_shape) == 4
            and input_shape[1] == IMAGE_SIZE
            and input_shape[2] == IMAGE_SIZE
            and input_shape[3] == 1
        ):
            return normalized_image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

        return normalized_image.reshape(1, IMAGE_SIZE * IMAGE_SIZE)

    def _prepare_cnn_input(self, normalized_image: np.ndarray) -> np.ndarray:
        return normalized_image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

    def predict_digit(self) -> None:
        if np.count_nonzero(self.image_array) == 0:
            self.ann_digit_var.set("-")
            self.ann_conf_var.set("--.-%")
            self.cnn_digit_var.set("-")
            self.cnn_conf_var.set("--.-%")
            self.verdict_label.config(text="Verdict: Draw something first", fg="#444444")
            return

        normalized_image = self.preprocess_for_mnist()

        ann_input = self._prepare_ann_input(normalized_image)
        cnn_input = self._prepare_cnn_input(normalized_image)

        ann_probs = self.ann_model.predict(ann_input, verbose=0)
        cnn_probs = self.cnn_model.predict(cnn_input, verbose=0)

        ann_digit = int(np.argmax(ann_probs, axis=1)[0])
        cnn_digit = int(np.argmax(cnn_probs, axis=1)[0])
        ann_conf = float(np.max(ann_probs)) * 100.0
        cnn_conf = float(np.max(cnn_probs)) * 100.0

        self.ann_digit_var.set(str(ann_digit))
        self.ann_conf_var.set(f"{ann_conf:04.1f}%")
        self.cnn_digit_var.set(str(cnn_digit))
        self.cnn_conf_var.set(f"{cnn_conf:04.1f}%")

        # Verdict logic
        verdict = "-"
        fg = "#444444"
        if ann_conf < 50 and cnn_conf < 50:
            verdict = "Low Confidence: Try again"
            fg = "#ff5555"
        elif ann_digit == cnn_digit:
            if ann_conf > cnn_conf:
                verdict = f"Match: ANN more confident ({ann_digit})"
                fg = "#28ac23"
            elif cnn_conf > ann_conf:
                verdict = f"Match: CNN more confident ({cnn_digit})"
                fg = "#00aaff"
            else:
                verdict = f"Match: Both equal ({ann_digit})"
                fg = "#444444"
        else:
            if cnn_conf > ann_conf:
                verdict = f"CNN Superior: {cnn_digit} ({cnn_conf:04.1f}%)"
                fg = "#00aaff"
            else:
                verdict = f"ANN Superior: {ann_digit} ({ann_conf:04.1f}%)"
                fg = "#28ac23"
        self.verdict_label.config(text=f"Verdict: {verdict}", fg=fg)

    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.image_array.fill(0)
        self.last_x, self.last_y = None, None
        self.ann_digit_var.set("-")
        self.ann_conf_var.set("--.-%")
        self.cnn_digit_var.set("-")
        self.cnn_conf_var.set("--.-%")
        self.verdict_label.config(text="Verdict: -", fg="#444444")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tkinter GUI for side-by-side ANN and CNN prediction.")
    parser.add_argument(
        "--ann-model-path",
        default="mnist_ann.keras",
        help="Path to trained ANN model (.keras or .h5).",
    )
    parser.add_argument(
        "--cnn-model-path",
        default="mnist_cnn.keras",
        help="Path to trained CNN model (.keras or .h5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = tk.Tk()
    # Center the window on the screen
    window_width = CANVAS_SIZE + 60
    window_height = CANVAS_SIZE + 220
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = int((screen_width / 2) - (window_width / 2))
    y = int((screen_height / 2) - (window_height / 2))
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    try:
        app = DigitRecognizerApp(
            root,
            ann_model_path=args.ann_model_path,
            cnn_model_path=args.cnn_model_path,
        )
    except Exception as exc:
        messagebox.showerror("Startup Error", str(exc))
        root.destroy()
        return

    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
