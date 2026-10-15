# TERRASCAN-AI // AI-Powered Camouflage & Anomaly Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-9acd32.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-9acd32.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Torchvision-ee4c2c.svg)](https://pytorch.org/)
[![Detector: YOLOv8](https://img.shields.io/badge/Detector-YOLOv8%20Ultralytics-00ffff.svg)](https://github.com/ultralytics/ultralytics)
[![UI: Tactical SITREP](https://img.shields.io/badge/UI-Tactical%20HUD%20Console-384525.svg)](#tactical-operations-console)

**TERRASCAN-AI** is a tactical intelligence and multi-spectral anomaly detection system engineered to intercept concealed structures, anomalous terrain disruptions, and camouflaged assets across complex visual environments (dense jungle canopies, aerial reconnaissance imagery, arid scrublands, and contested perimeters).

---

```
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
```

---

## 1. Core Architecture & Capabilities

### A. Camouflage Intercept Engine (`terrascan/camouflage.py`)
- **Hou & Zhang Spectral Residual Saliency**: Computes Fourier log-spectrum disparity $R(f) = L(f) - h_n(f) * L(f)$ to highlight high-frequency anomalies across natural background distributions.
- **Deep Patch Clustering**: Unsupervised $k$-means feature clustering ($k=4$) separating foreground disruptive textures from ambient terrain.
- **Color Opponent Contrast**: Multi-scale CIE-LAB color disparity highlighting non-foliage pigment signatures.
- **Border Margin Tapering**: Cosine-tapered border weighting to eliminate 2D FFT edge ringing and spurious border false positives.

### B. PatchCore Anomaly Detector (`terrascan/anomaly.py`)
- **Deep Feature Representation**: Extracts spatial patch representations from WideResNet-50 (`layer2` + `layer3`) with spatial neighborhood context pooling.
- **Dual-Operational Modes**:
  - **Self-Reference Mode (Zero-Shot)**: Scores visual outliers against internal scene patch density within a single uploaded image.
  - **Reference Memory Mode**: Matches patches against nominal terrain memory banks stored in `data/reference/`.

### C. Tactical Object Recognition (`terrascan/objects.py`)
- **YOLOv8n Neural Detector**: Real-time object localization for recognizable military and transport assets (vehicles, aircraft, personnel, trucks).
- **Graceful Fallback**: Continues uninterrupted with simulated tactical detection heuristics if weights or network connectivity are unavailable.

### D. Multi-Sensor Fusion & Threat Index (`terrascan/fusion.py`)
- **Co-Activation Synthesis**: Combines camouflage and anomaly evidence with cross-modal co-activation reinforcement bonuses.
- **False-Positive Suppression**: Aspect ratio filtering ($>4.0$ strips rejected) and minimum confidence gating.
- **Calibrated Threat Index (0–100)**: Heuristic threat assessment score mapped to alert levels (`LOW`, `MODERATE`, `HIGH CONCERN`, `CRITICAL`).

---

## 2. Directory Structure

```
terascan-2.0/
├── app.py                      # Flask application & REST API server
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
├── LICENSE                     # MIT Open Source License
├── .gitignore                  # Git exclusion rules
├── .gitattributes              # Line-ending & binary attributes
├── data/
│   └── reference/              # Memory bank storage for reference terrain
│       └── .gitkeep
├── models/
│   └── .gitkeep                # Storage for optional trained COD checkpoints
├── samples/                    # Synthetic terrain benchmark datasets
│   ├── sample_01.png           # Dense foliage with camouflaged bunker
│   ├── sample_02.png           # Aerial forest canopy with geometric clearing
│   └── sample_03.png           # Arid scrubland with netting & vehicle
├── templates/
│   └── index.html              # Tactical SITREP HUD frontend
├── terrascan/
│   ├── __init__.py             # Package declaration
│   ├── anomaly.py              # PatchCore-Lite anomaly detection engine
│   ├── camouflage.py           # Spectral residual & clustering camo engine
│   ├── fusion.py               # Fused suspicion map & threat index generator
│   ├── objects.py              # YOLOv8 object recognition engine
│   ├── pipeline.py             # Multi-sensor orchestrator
│   ├── samples_generator.py    # Synthetic terrain generator
│   └── utils.py                # Color-mapping, base64, & image utilities
└── tests/
    └── test_pipeline.py        # Automated test suite (17 unit & integration tests)
```

---

## 3. Quickstart & Installation

### Requirements
- **Python**: 3.10 or higher (tested on Python 3.10 – 3.14 on Linux, macOS, and Windows)
- **Hardware**: CPU or CUDA-compatible GPU (auto-detected)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/priyanshu-kuntal/TERRASCAN-AI.git
cd TERRASCAN-AI

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the Tactical Mission Server
python app.py
```

Open your browser to `http://127.0.0.1:5000` to interact with the Tactical Operations Console.

---

## 4. Tactical Operations Console

The web UI is designed as a military SITREP (Situation Report) operations room HUD:
- **Interactive Before/After Split Slider**: Compare raw sensor inputs against active suspicion overlays.
- **Full Viewport Crosshair Tracking**: Smooth targeting hairlines displaying coordinates and local anomaly scores.
- **Sector Marker Pins**: Interactive pins with animated sonar ping effects linked directly to the SITREP table.
- **Calibrated Threat Index Gauge**: Semi-circular graduated meter with radial ticks (0–100) and alert badges.
- **Client-Side Dossier Export**: Generate high-resolution classified mission intelligence reports in PNG format using the HTML5 Canvas engine.

### Keyboard Shortcuts
| Key | Action |
|:---:|:---|
| <kbd>R</kbd> | Run Tactical Scan on active imagery |
| <kbd>S</kbd> | Toggle Before/After Split Comparison Slider |
| <kbd>1</kbd> | View Raw Base Sensor Imagery |
| <kbd>2</kbd> | View Camouflage Detection Overlay |
| <kbd>3</kbd> | View PatchCore Anomaly Heatmap |
| <kbd>4</kbd> | View YOLOv8 Object Contact Boxes |
| <kbd>5</kbd> | View Fused Suspicion Heatmap |
| <kbd>6</kbd> | View All Layers Composite |
| <kbd>ESC</kbd> | Clear & Reset Tactical Viewport |

---

## 5. RESTful API Reference

### `GET /api/status`
Returns system telemetry, hardware device acceleration, and active engine states:
```json
{
  "status": "OPERATIONAL",
  "uptime_seconds": 240,
  "system": {
    "device": "CPU",
    "camo_engine": "RESIDUAL+LAB",
    "anomaly_mode": "SELF-REFERENCE",
    "reference_count": 0,
    "yolo_status": "ONLINE",
    "yolo_online": true
  }
}
```

### `POST /api/analyze`
Executes multi-spectral analysis on uploaded imagery or sample payload:
```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -F "image=@aerial_recon.jpg" \
  -F "camo_threshold=0.50" \
  -F "camo_weight=0.60" \
  -F "anomaly_weight=0.40" \
  -F "object_conf=0.25" \
  -F "overlay_opacity=0.60" \
  -F "enable_camo=true" \
  -F "enable_anomaly=true" \
  -F "enable_objects=true"
```

### `POST /api/reference`
Ingest nominal terrain imagery into the anomaly memory bank:
```bash
curl -X POST http://127.0.0.1:5000/api/reference \
  -F "reference_images=@nominal_terrain_01.png" \
  -F "reference_images=@nominal_terrain_02.png"
```

### `DELETE /api/reference`
Purge all stored reference terrain data and revert to zero-shot self-reference mode.

---

## 6. Running Tests

Run the complete 17-test test suite using `pytest`:

```bash
python -m pytest tests/test_pipeline.py -v
```

All 17 tests validate:
- Camouflage saliency normalization, clustering, and edge suppression
- PatchCore feature extraction and distance scoring
- YOLO object detection and fallback resilience
- Weighted fusion, bounding box ranking, and threat indexing
- REST API response structures and edge-case handling

---

## 7. License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for full details.

Developed by **[Priyanshu Kuntal](https://github.com/priyanshu-kuntal)**.
