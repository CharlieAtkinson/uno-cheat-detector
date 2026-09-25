# 🃏 Autonomous UNO Card Game Cheating Detection Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11_Nano-green.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end computer vision and rule-verification system that analyzes video streams of UNO games, tracks played cards at ~12–15 FPS on a consumer CPU, and automatically flags illegal moves in real time.

![System Demo](assets/demo.gif)

---

## 🎯 System Highlights

- **Edge-Optimized Detection:** Fine-tuned YOLOv11 Nano on an 11-class UNO deck subset, reaching **0.970 mAP@50** and **0.949 Precision** at 640px resolution.
- **Occlusion & Hand-Noise Rejection:** Dual-stage **Temporal Debounce Filter** (3-frame stability window + 5-frame grace period) eliminates false detections caused by card placement gestures and hand transit.
- **Deterministic Rule Engine:** LIFO state machine tracking color, number, and wildcard precedence to detect cheating without cloud or GPU dependencies.
- **CPU-First Execution:** Inference pipeline delivering **~12 FPS on an 11th Gen Intel i5 CPU** without requiring a dedicated GPU.

---

## 🏗️ Architecture & Pipeline

```text
[ Video Stream ]
       │
       ▼
[ YOLOv11n Detection (imgsz=640, conf=0.50) ]
       │
       ▼
[ Temporal Debounce Filter ] ── (Rejects transient occlusion & hand transit)
       │ (Candidate locked after 3 consecutive stable frames)
       ▼
[ LIFO Game State Machine ]
       │ ── Color Match?
       │ ── Value / Action Match?
       │ ── Wildcard Precedence?
       ▼
[ Decision Logger: VALID (1) / CHEAT DETECTED (0) ]
```

---

## 📊 Model Evaluation & Metrics

The model was trained on custom-annotated imagery across diverse angles, lighting conditions, and partial overlaps using aggressive spatial and color augmentations:
- **Spatial Augmentations:** `degrees=30.0`, `perspective=0.0001`, `mosaic=1.0`, `mixup=0.1` (forcing the network to recognize central card symbols during heavy occlusions).
- **Photometric Augmentations:** HSV color jittering to preserve clean red/green decision boundaries under harsh shadows and color shifts.

| Metric | Score | Note |
|---|---|---|
| **mAP@50** | **0.9702** | Peak convergence at Epoch 104 |
| **Precision** | **0.9490** | Minimal ghost detections |
| **Recall** | **0.9146** | Robust detection under acute angles |
| **Color Accuracy** | **100%** | Zero confusion between Red and Green classes |

<p align="center">
  <img src="assets/metrics/confusion_matrix.png" alt="Confusion Matrix" width="48%">
  <img src="assets/metrics/training_curves.png" alt="Training Curves" width="48%">
</p>

> **Edge-Case Insight:** Rapid card placement created occasional motion blur between `red_0` and `red_skip`. The 3-frame temporal debounce filter neutralizes this by deferring state updates until motion stops on the play stack.

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone [https://github.com/CharlieAtkinson/uno-cheat-detector.git](https://github.com/CharlieAtkinson/uno-cheat-detector.git)
cd uno-cheat-detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Model Weights
Download `weights.pt` from the [v1.0.0 Release](https://github.com/CharlieAtkinson/uno-cheat-detector/releases/tag/v1.0.0) and place it inside the `weights/` directory.

### 3. Run Inference
Process a video file to output the play audit trail:
```bash
python main.py data/sample_game.mp4
```

Run with visual bounding boxes and live detection output:
```bash
python main.py data/sample_game.mp4 --debug
```

### 4. Output Specification
The system exports an audit trail (`output.txt`):
```csv
sample_game.mp4,1,red_0,32,1.07,1
sample_game.mp4,7,green_0,85,2.83,1
sample_game.mp4,8,green_1,142,4.73,1
sample_game.mp4,9,green_2,210,7.00,0
```
*(Format: `filename, card_class_id, card_name, first_seen_frame, timestamp_sec, validity`)*

---

## 🧪 Automated Testing
Execute unit tests for rule transitions and edge cases:
```bash
pytest tests/
```

---

## 🛠️ Tech Stack
- **Computer Vision:** Ultralytics YOLOv11 Nano, OpenCV
- **Software Architecture:** Python 3.10+, Pytest, Dataclasses, Argparse