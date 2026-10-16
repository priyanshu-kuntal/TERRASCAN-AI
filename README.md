# TERRASCAN-AI // AI-Powered Camouflage & Anomaly Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-9acd32.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-9acd32.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Torchvision-ee4c2c.svg)](https://pytorch.org/)
[![Detector: YOLOv8](https://img.shields.io/badge/Detector-YOLOv8%20Ultralytics-00ffff.svg)](https://github.com/ultralytics/ultralytics)
[![UI: Tactical SITREP](https://img.shields.io/badge/UI-Tactical%20HUD%20Console-384525.svg)](#4-tactical-operations-console)
[![Docker: Ready](https://img.shields.io/badge/Docker-Ready-blue.svg)](#docker-deployment)

**TERRASCAN-AI** is a tactical intelligence and multi-spectral anomaly detection system engineered to intercept concealed structures, anomalous terrain disruptions, and camouflaged assets across complex visual environments (dense jungle canopies, aerial reconnaissance imagery, arid scrublands, and contested perimeters).

---

## System Overview & Pipeline Architecture



+-----------------------------------------------------------------------------------------+
|                                    TERRASCAN-AI PIPELINE                                |
+-----------------------------------------------------------------------------------------+
|
[ Ingest Image ]
|
+-------------------------+-------------------------+
|                                                   |
v                                                   v
[ Camouflage Detector ]                              [ Anomaly Detector ]
(Spectral Residual Saliency                           (PatchCore-Lite WideResNet
+ Deep Patch Clustering                              Layer2+3 Patch Tokens +
+ LAB Surround Contrast)                              Ref / Self-Ref Outlier kNN)
|                                                   |
|                                                   v
|                                          [ Object Detector ]
|                                           (YOLOv8 Tactical)
|                                                   |
+-------------------------+-------------------------+
|
v
[ Multi-Sensor Fusion ]
suspicion = (0.6 * camo) + (0.4 * anomaly) + bonus
|
+-------------------------+-------------------------+
|                                                   |
v                                                   v
[ Top 5 Ranked Regions ]                             [ Threat Index 0-100 ]
(ALPHA, BRAVO, CHARLIE, ...)                          (Calibrated Risk Gauge)
|                                                   |
+-------------------------+-------------------------+
|
v
[ Tactical Overlays & Briefing ]
(Original, Camo, Anomaly, YOLO, Fused)


---

## 1. Core Architecture & Detection Engines

### A. Camouflage Intercept Engine (`terrascan/camouflage.py`)
* **Hou & Zhang Spectral Residual Saliency**: Computes the Fourier log-spectrum disparity to highlight high-frequency anomalies across natural background distributions:
  $$R(f) = L(f) - h_n(f) * L(f)$$
  where $L(f)$ represents the log spectrum of the image and $h_n(f)$ is an $n \times n$ spatial averaging filter.
* **Deep Patch Clustering**: Unsupervised $k$-means feature clustering ($k=4$) separating foreground disruptive textures from ambient terrain patterns.
* **Color Opponent Contrast**: Multi-scale CIE-LAB color disparity highlighting non-foliage pigment signatures and unnatural synthetic coatings.
* **Border Margin Tapering**: Cosine-tapered boundary weighting eliminating 2D FFT edge ringing and spurious perimeter artifacts.

### B. PatchCore Anomaly Detector (`terrascan/anomaly.py`)
* **Deep Feature Representation**: Extracts spatial patch representations from WideResNet-50 (`layer2` + `layer3`) with spatial neighborhood context pooling.
* **Dual-Operational Modes**:
  * **Self-Reference Mode (Zero-Shot)**: Computes visual outlier distance against internal scene patch density within a single ingested tile.
  * **Reference Memory Bank Mode**: Matches candidate patches against nominal terrain memory banks stored in `data/reference/`.

### C. Tactical Object Recognition (`terrascan/objects.py`)
* **YOLOv8n Neural Detector**: Real-time localization for military, transport, and personnel signatures (vehicles, aircraft, vessels, defensive structures).
* **Resilient Fallback**: Gracefully transitions to simulated tactical heuristics if model weights or GPU backends are offline.

### D. Multi-Sensor Fusion & Threat Calibration (`terrascan/fusion.py`)
* **Co-Activation Synthesis**: Dynamically weights camouflage and anomaly activations with cross-modal reinforcement bonuses:
  $$\text{Score}_{\text{fused}} = (w_c \cdot S_{\text{camo}}) + (w_a \cdot S_{\text{anomaly}}) + \Delta_{\text{bonus}}$$
* **False-Positive Suppression**: Aspect ratio geometric gating ($>4.0$ linear strip rejections) and minimum confidence clustering.
* **Threat Index (0–100)**: Quantized composite risk gauge mapped to four operational alert states:
  * `00 – 25`: **LOW** (Baseline ambient terrain)
  * `26 – 50`: **MODERATE** (Minor structural or pigment anomaly)
  * `51 – 75`: **HIGH CONCERN** (Co-activated camouflage & deep feature outlier)
  * `76 – 100`: **CRITICAL** (Confirmed vehicle/structure contact in concealed sector)

---

## 2. Directory Structure



TERRASCAN-AI/
├── app.py                      # Flask application & REST API server
├── Dockerfile                  # Containerized deployment definition
├── requirements.txt            # Python runtime dependencies
├── README.md                   # System documentation
├── LICENSE                     # MIT Open Source License
├── .gitignore                  # Git exclusion rules
├── .gitattributes              # Line-ending & binary attributes
├── data/
│   └── reference/              # Memory bank storage for reference terrain
│       └── .gitkeep
├── models/
│   └── .gitkeep                # Storage for trained COD / YOLO checkpoints
├── samples/                    # Synthetic terrain benchmark datasets
│   ├── sample_01.png           # Dense foliage with camouflaged bunker
│   ├── sample_02.png           # Aerial forest canopy with geometric clearing
│   └── sample_03.png           # Arid scrubland with netting & vehicle
├── templates/
│   └── index.html              # Tactical SITREP HUD frontend
├── terrascan/
│   ├── init.py             # Package declaration
│   ├── anomaly.py              # PatchCore-Lite anomaly detection engine
│   ├── camouflage.py           # Spectral residual & clustering camo engine
│   ├── fusion.py               # Fused suspicion map & threat index generator
│   ├── objects.py              # YOLOv8 object recognition engine
│   ├── pipeline.py             # Multi-sensor orchestrator
│   ├── samples_generator.py    # Synthetic terrain generator
│   └── utils.py                # Color-mapping, base64, & image utilities
└── tests/
└── test_pipeline.py        # Automated test suite (17 unit & integration tests)


---

## 3. Quickstart & Installation

### Requirements
* **Python**: 3.10 or higher (tested on 3.10, 3.11, 3.12)
* **Hardware**: CPU or CUDA-capable GPU (auto-detected via `torch.cuda.is_available()`)

### Standard Setup

```bash
# 1. Clone the repository
git clone https://github.com/priyanshu-kuntal/TERRASCAN-AI.git
cd TERRASCAN-AI

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Tactical Mission Server
python app.py


Navigate to http://127.0.0.1:5000 to interact with the Tactical Operations Console.

Docker Deployment

# Build the container
docker build -t terrascan-ai:latest .

# Run with port forwarding
docker run -p 5000:5000 --rm terrascan-ai:latest


4. Tactical Operations Console

The web interface operates as an electronic Situation Report (SITREP) mission console:

Interactive Split Comparison Slider: Real-time sliding wipe comparing raw sensor imagery against active threat overlays.

Viewport Crosshair Targeting: Dynamic mouse cursor tracking with localized anomaly index readout.

Sector Markers (ALPHA through ECHO): Sonar-animated sector pins linked directly to the threat classification table.

Graduated Threat Index Gauge: Circular radial meter calibrated 0–100 with dynamic color grading.

Classified Dossier Export: One-click generation of exportable PNG mission briefing dossiers via HTML5 Canvas.

Keyboard Shortcuts

Key

Action

<kbd>R</kbd>

Run Tactical Analysis on active tile

<kbd>S</kbd>

Toggle Before/After Split Comparison Slider

<kbd>1</kbd>

Display Raw Base Sensor Imagery

<kbd>2</kbd>

Display Camouflage Detection Layer

<kbd>3</kbd>

Display PatchCore Anomaly Heatmap

<kbd>4</kbd>

Display YOLOv8 Object Contact Bounding Boxes

<kbd>5</kbd>

Display Fused Suspicion Heatmap

<kbd>6</kbd>

Display Multi-Layer Composite View

<kbd>ESC</kbd>

Clear & Reset Tactical Viewport

5. Performance Benchmarks

Inference evaluated on standard  aerial reconnaissance tiles:

Component / Layer

Intel Core i7 (CPU)

Apple M2 / M3 (MPS)

NVIDIA RTX 3060 / T4

Peak VRAM

Spectral Saliency (FFT + LAB)

48 ms

22 ms

14 ms

~180 MB

PatchCore-Lite (WideResNet)

310 ms

98 ms

41 ms

~740 MB

YOLOv8n Neural Localization

65 ms

24 ms

12 ms

~260 MB

Fusion & Sector Ranking

18 ms

8 ms

5 ms

< 50 MB

Full Pipeline Total

~441 ms

~152 ms

~72 ms

~1.1 GB

6. RESTful API Reference

GET /api/status

Returns telemetry, hardware acceleration status, and active engine configurations.

{
  "status": "OPERATIONAL",
  "uptime_seconds": 240,
  "system": {
    "device": "CUDA",
    "camo_engine": "RESIDUAL+LAB",
    "anomaly_mode": "SELF-REFERENCE",
    "reference_count": 0,
    "yolo_status": "ONLINE",
    "yolo_online": true
  }
}


POST /api/analyze

Executes multi-spectral analysis on an uploaded image tile.

curl -X POST http://127.0.0.1:5000/api/analyze \
  -F "image=@samples/sample_01.png" \
  -F "camo_threshold=0.50" \
  -F "camo_weight=0.60" \
  -F "anomaly_weight=0.40" \
  -F "object_conf=0.25" \
  -F "overlay_opacity=0.60" \
  -F "enable_camo=true" \
  -F "enable_anomaly=true" \
  -F "enable_objects=true"


POST /api/reference

Ingest nominal terrain baselines into the PatchCore memory bank.

curl -X POST http://127.0.0.1:5000/api/reference \
  -F "reference_images=@nominal_terrain_01.png" \
  -F "reference_images=@nominal_terrain_02.png"


DELETE /api/reference

Purges memory bank storage and reverts the anomaly engine to zero-shot self-reference mode.

7. Automated Test Suite

Run the full 17-test verification suite using pytest:

python -m pytest tests/test_pipeline.py -v


Coverage includes:

FFT spectral residual calculation, numerical stability, and cosine border tapering.

PatchCore neighborhood token extraction and distance matrix normalization.

YOLOv8 weight loading and headless fallback handling.

Aspect ratio filtering (), bounding box non-maximum clustering, and threat scoring.

REST API endpoints, multipart upload validation, and payload integrity.

8. Academic References

Spectral Residual Saliency: Hou, X., & Zhang, L. (2007). Saliency Detection: A Spectral Residual Approach. IEEE CVPR.

PatchCore Anomaly Detection: Roth, K., et al. (2022). Towards Total Recall in Industrial Anomaly Detection. IEEE CVPR.

YOLOv8: Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 Architecture. Ultralytics Inc.
