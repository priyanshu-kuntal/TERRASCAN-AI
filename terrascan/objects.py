"""
TERRASCAN-AI — Object Detection Engine
Ultralytics YOLOv8 wrapper with lazy-loading, auto-download,
and graceful offline/failure fallback.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Tactical Object Detector wrapping YOLOv8n.
    Resilient: If weights fail to download or PyTorch/YOLO has issues,
    it marks the engine as OFFLINE and continues without crashing.
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model_name = model_name
        self._model = None
        self._initialized = False
        self._available = False
        self._error_msg: Optional[str] = None

    def _load_model(self) -> bool:
        """Attempt lazy loading of Ultralytics YOLO model."""
        if self._initialized:
            return self._available

        self._initialized = True
        try:
            from ultralytics import YOLO
            # YOLO auto-downloads yolov8n.pt if not locally present
            self._model = YOLO(self.model_name)
            self._available = True
            logger.info("YOLOv8 detector successfully loaded.")
            return True
        except Exception as e:
            self._model = None
            self._available = False
            self._error_msg = str(e)
            logger.warning("YOLOv8 loading failed: %s. Running in OFFLINE mode.", e)
            return False

    def is_available(self) -> bool:
        """Check if YOLO model is ready for inference."""
        if not self._initialized:
            self._load_model()
        return self._available

    def get_status(self) -> Dict[str, Any]:
        """Return status dictionary for UI chips."""
        ready = self.is_available()
        return {
            "online": ready,
            "status_label": "ONLINE" if ready else "OFFLINE",
            "model_name": self.model_name,
            "error": self._error_msg,
        }

    def detect(
        self, img_rgb: np.ndarray, conf_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Execute object detection on an RGB image.

        Args:
            img_rgb: np.ndarray (H, W, 3) in [0, 255]
            conf_threshold: minimum confidence score [0.05, 0.95]

        Returns:
            List of detected objects:
            [
                {
                    "box": [x1, y1, x2, y2],
                    "label": "person",
                    "score": 0.87,
                    "class_id": 0
                },
                ...
            ]
        """
        if not self.is_available() or self._model is None:
            return []

        h, w = img_rgb.shape[:2]
        detections: List[Dict[str, Any]] = []

        try:
            # Run inference; verbose=False to keep terminal logs clean
            results = self._model.predict(
                source=img_rgb,
                conf=float(conf_threshold),
                verbose=False,
                device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
            )

            if results and len(results) > 0:
                res = results[0]
                boxes = res.boxes
                if boxes is not None:
                    names = res.names or {}
                    for box in boxes:
                        coords = box.xyxy[0].cpu().numpy().tolist()
                        x1, y1, x2, y2 = [int(round(c)) for c in coords]
                        # Clamp to image bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        score = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = names.get(cls_id, f"class_{cls_id}")

                        detections.append(
                            {
                                "box": [x1, y1, x2, y2],
                                "label": str(cls_name),
                                "score": round(score, 3),
                                "class_id": cls_id,
                            }
                        )
        except Exception as e:
            logger.error("Object detection error during inference: %s", e)
            return []

        return detections
