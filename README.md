# Car Damage — YOLO26 detection

Academic project for **detecting car damage** in images using **Ultralytics YOLO26** (nano variant). The workflow downloads a labeled dataset from **Roboflow**, trains a detector, and ships the resulting weights plus standard training visualizations.

## What’s in this repository

| Item | Description |
|------|-------------|
| `CarDamageYOLO26.ipynb` | End-to-end notebook: install deps, pull dataset, train with CLI |
| `car_damage_26n.pt` | Trained **YOLO26n** weights (object detection checkpoint) |
| `results.png` | Training loss / metrics overview |
| `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png` | Precision, recall, and F1 curves for box detection |
| `confusion_matrix.png`, `confusion_matrix_normalized.png` | Confusion matrices on the validation split |
| `labels.jpg` | Label distribution / dataset summary plot from training |

## Tech stack

- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** — training and inference (`yolo` CLI and Python API)
- **[Roboflow](https://roboflow.com/)** — dataset versioning and export in **YOLO26** format
- **PyTorch** (with CUDA when available) — backend used by Ultralytics during training

Training in the reference run used **50 epochs**, image size **640**, batch **64**, starting from **`yolo26n.pt`**, on a dataset exported as `Car-Damage-Detection-1` with `data.yaml` (paths in the notebook target Google Colab under `/content/`).

## Setup

```bash
pip install ultralytics roboflow
```

For GPU training, install a **CUDA-enabled** PyTorch build that matches your system; see the [PyTorch](https://pytorch.org/get-started/locally/) and [Ultralytics](https://docs.ultralytics.com/quickstart/) docs.

## Reproducing training

1. Open `CarDamageYOLO26.ipynb` in **Jupyter**, **VS Code**, or **Google Colab**.
2. **Dataset**: the notebook uses the Roboflow Python SDK to download project version `1` in **yolo26** format.  
   - **Do not commit API keys.** Prefer an environment variable, for example:
     ```python
     import os
     from roboflow import Roboflow
     rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
     ```
   - After download, point `data=` in the `yolo train` command to your local `data.yaml` (on Colab this was under `/content/Car-Damage-Detection-1/data.yaml`).
3. **Train** (adjust paths for your machine):
   ```bash
   yolo train model=yolo26n.pt data=path/to/data.yaml epochs=50 imgsz=640 batch=64
   ```
   Lower `batch` if you run out of VRAM.

If your Roboflow API key was ever committed in the notebook, **rotate it** in the Roboflow dashboard and use env-based configuration going forward.

## Inference with the bundled weights

From the repo root (Python):

```python
from ultralytics import YOLO

model = YOLO("car_damage_26n.pt")
results = model.predict("your_image.jpg", save=True, conf=0.25)
```

Or CLI:

```bash
yolo predict model=car_damage_26n.pt source=your_image.jpg
```

## Results

Interpret the PNGs in the root folder together with Ultralytics run logs (under `runs/detect/` when training locally). They summarize convergence, per-class behavior, and validation performance for the exported model.

## Credits

- Dataset and labeling workflow via **Roboflow** (project: car damage detection, workspace as configured in the notebook).
- Model architecture and training pipeline: **Ultralytics YOLO26**.

## License

Weights and dataset may be subject to **third-party terms** (Ultralytics, Roboflow, and the original dataset license). Confirm usage rights before redistribution or commercial use.

---

*Course / research context: CarDamage — vehicle damage object detection with YOLO26n.*
