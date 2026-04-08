# Midterm Project: Handwritten Digit Recognition (ANN and CNN)

This repository is organized for rubric-focused grading and demo.

## Grading-Focused Files

- `train_mnist_models.py`: trains ANN and CNN, evaluates both, prints per-digit accuracy, and saves both models.
- `digit_draw_app.py`: minimalist Tkinter + OpenCV drawing app with side-by-side ANN/CNN predictions.
- `mnist_ann.keras`: trained ANN model.
- `mnist_cnn.keras`: trained CNN model.
- `MIDTERM_DOCUMENTATION.txt`: rubric mapping, implementation notes, and test evidence.
- `requirements.txt`: required Python packages for training and GUI.

## Archived Non-Grading Files

Legacy CSV-based scripts and artifacts were moved to keep the root clean:

- `archive_not_for_grading/legacy_csv_pipeline/`

These files are not part of the final rubric demo workflow.

## 1. Setup

Use Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train ANN and CNN

```bash
python train_mnist_models.py --epochs 10 --batch-size 128 --seed 42 --validation-split 0.1 --ann-model-path mnist_ann.keras --cnn-model-path mnist_cnn.keras
```

This performs:

- MNIST download,
- normalization,
- ANN/CNN reshaping,
- one-hot encoding,
- model training and evaluation,
- per-digit accuracy reporting,
- saving ANN and CNN model files.

## 3. Launch the GUI (Side-by-Side Comparison)

```bash
python digit_draw_app.py --ann-model-path mnist_ann.keras --cnn-model-path mnist_cnn.keras
```

GUI features:

- white drawing canvas,
- Predict and Clear buttons,
- side-by-side ANN and CNN predicted digit with confidence,
- verdict status bar:
  - Match (both agree),
  - CNN Superior or ANN Superior (higher confidence),
  - Low Confidence (both below 50%),
- centered startup window for cleaner demo presentation.

## Recommended Demo Flow

1. Train both models.
2. Launch the drawing app.
3. Draw multiple digits and show ANN vs CNN verdict behavior.
4. Capture screenshot(s) for `MIDTERM_DOCUMENTATION.txt` where indicated.

## Notes

- If TensorFlow installation fails, run:

```bash
python -m pip install --upgrade pip
```
