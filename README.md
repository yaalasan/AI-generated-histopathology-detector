# Histopathology AI Image Detector

A deep learning classifier that distinguishes **real histopathology images** from **AI-generated fakes**, using transfer learning on ResNet-18.

---

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | 98.75% |
| Test AUC | 0.9977 |
| False Positives (real misclassified as fake) | 0 |
| False Negatives (fake misclassified as real) | 1 |

**Training curves and confusion matrix:**

| Training Loss | Validation Accuracy | Confusion Matrix |
|---|---|---|
| ![loss](https://raw.githubusercontent.com/yaalasan/AI-generated-histopathology-detector/main/assets/training_loss.png) | ![loss](https://raw.githubusercontent.com/yaalasan/AI-generated-histopathology-detector/main/assets/training_loss.png) | ![loss](https://raw.githubusercontent.com/yaalasan/AI-generated-histopathology-detector/main/assets/training_loss.png) |

---

## Project Structure

```
histopathology-detector/
|
|-- configs/
|   +-- config.yaml          # All hyperparameters in one place
|
|-- src/                     # Core library (importable modules)
|   |-- config.py            # Config loader
|   |-- dataset.py           # Transforms, splits, DataLoaders
|   |-- model.py             # ResNet builder
|   |-- train.py             # Training loop, early stopping, checkpointing
|   |-- evaluate.py          # Metrics, confusion matrix, ROC curve
|   +-- predict.py           # Single image and batch inference
|
|-- scripts/                 # Entry points (run these)
|   |-- train.py             # Run training
|   |-- evaluate.py          # Run test-set evaluation
|   +-- predict.py           # Run inference on images
|
|-- tests/
|   +-- test_model.py        # Sanity checks (pytest)
|
|-- assets/                  # Saved plots for README
|-- dataset/
|   |-- fake/                # AI-generated histopathology images
|   +-- real/                # Real histopathology images
|-- outputs/                 # Model weights and result plots (git-ignored)
|-- requirements.txt
+-- README.md
```

---

## Setup

```bash
git clone https://github.com/yourusername/histopathology-detector.git
cd histopathology-detector

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Dataset

Organise your images like this:

```
dataset/
  fake/
    image001.png
    image002.png
    ...
  real/
    image001.png
    image002.png
    ...
```

The dataset is automatically split **70% train / 15% val / 15% test** with a fixed seed for reproducibility.

---

## Usage

### Train

```bash
python scripts/train.py
# or with a custom config:
python scripts/train.py --config configs/config.yaml
```

Training features:
- Data augmentation (flips, rotation, colour jitter)
- Early stopping (default patience: 5 epochs)
- Best-model checkpointing (saves `outputs/best_model.pth`)
- LR scheduler (ReduceLROnPlateau)

### Evaluate on Test Set

```bash
python scripts/evaluate.py
```

Outputs saved to `outputs/plots/`:
- `confusion_matrix.png`
- `roc_curve.png`
- `training_curves.png`
- `classification_report.txt`

### Predict on New Images

```bash
# Single image
python scripts/predict.py --image path/to/image.png

# Entire folder
python scripts/predict.py --folder path/to/folder/
```

Example output:
```
  Image      : sample.png
  Prediction : FAKE
  Confidence : 97.3%
  Probabilities:
    fake: 97.3%
    real: 2.7%
```

---

## Configuration

All settings are in `configs/config.yaml`. Key options:

```yaml
training:
  epochs: 20
  learning_rate: 0.0001
  early_stopping_patience: 5

model:
  architecture: "resnet18"   # or resnet50
  freeze_backbone: true

augmentation:
  enabled: true
  horizontal_flip: true
  vertical_flip: true
  rotation_degrees: 15
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Model Architecture

- **Base**: ResNet-18 pre-trained on ImageNet
- **Backbone**: Frozen (transfer learning)
- **Head**: Single linear layer (512 → 2)
- **Optimiser**: Adam with ReduceLROnPlateau scheduler
- **Loss**: CrossEntropyLoss

---

## License

MIT
