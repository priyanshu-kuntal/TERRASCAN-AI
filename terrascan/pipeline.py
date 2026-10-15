"""
TERRASCAN-AI — Tactical Multi-Model Pipeline Orchestrator
Coordinates camouflage detection, PatchCore-lite anomaly detection,
YOLOv8 object recognition, and multi-sensor fusion with stage timing and logging.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union

import numpy as np

from terrascan.utils import (
    get_device,
    get_device_name,
    load_image,
    resize_image_for_inference,
    to_base64_png,
)
from terrascan.camouflage import CamouflageDetector
from terrascan.anomaly import AnomalyDetector
from terrascan.objects import ObjectDetector
from terrascan.fusion import FusionEngine


class TerraScanPipeline:
    """
    Main orchestration engine for TERRASCAN-AI.
    Initializes ML components and provides high-performance multi-modal analysis.
    """

    def __init__(
        self,
        models_dir: str = "models",
        reference_dir: str = "data/reference",
        yolo_weights: str = "yolov8n.pt",
    ):
        self.device = get_device()
        self.device_name = get_device_name()

        # Component detectors
        self.camo_detector = CamouflageDetector(models_dir=models_dir, device=self.device)
        self.anomaly_detector = AnomalyDetector(reference_dir=reference_dir, device=self.device)
        self.object_detector = ObjectDetector(model_name=yolo_weights)
        self.fusion_engine = FusionEngine()

    def get_system_status(self) -> Dict[str, Any]:
        """Return system health, device, and sub-engine operational modes."""
        anomaly_status = self.anomaly_detector.get_status()
        yolo_status = self.object_detector.get_status()

        return {
            "device": self.device_name,
            "camo_engine": self.camo_detector.engine_mode,
            "anomaly_mode": anomaly_status["mode"],
            "reference_count": anomaly_status["reference_count"],
            "yolo_status": yolo_status["status_label"],
            "yolo_online": yolo_status["online"],
        }

    def run_analysis(
        self,
        image_input: Any,
        camo_threshold: float = 0.50,
        camo_weight: float = 0.60,
        anomaly_weight: float = 0.40,
        object_conf: float = 0.25,
        overlay_opacity: float = 0.60,
        enable_camo: bool = True,
        enable_anomaly: bool = True,
        enable_objects: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute end-to-end tactical analysis on the provided image.

        Returns:
            Structured dictionary containing base64 overlays, detections, ranked regions,
            threat index, stage timings, and tactical execution logs.
        """
        total_start_time = time.perf_counter()
        logs: list[str] = []

        def add_log(msg: str) -> None:
            utc_now = datetime.now(timezone.utc).strftime("%H:%M:%S")
            logs.append(f"> {utc_now} {msg}")

        add_log("INGESTING SENSOR PAYLOAD...")

        # 1. Load and Standardize Image
        img_rgb = load_image(image_input)
        orig_h, orig_w = img_rgb.shape[:2]

        # 2. Resize for Inference if needed (long side <= 1280 px)
        infer_img, scale, _ = resize_image_for_inference(img_rgb, max_long_side=1280)
        ih, iw = infer_img.shape[:2]

        add_log(f"FRAME GEOMETRY: {orig_w}x{orig_h} (INFERENCE RES: {iw}x{ih}, SCALE: {scale:.2f})")

        # 3. Camouflage Detection
        camo_map = None
        camo_contours = []
        camo_regions_count = 0
        t_camo = 0.0

        if enable_camo:
            t0 = time.perf_counter()
            add_log(f"INITIATING CAMOUFLAGE SCAN [{self.camo_detector.engine_mode}]...")
            camo_res = self.camo_detector.detect(infer_img, threshold=camo_threshold)
            camo_map = camo_res["score_map"]
            camo_contours = camo_res["contours"]
            camo_regions_count = camo_res["regions_count"]
            t_camo = time.perf_counter() - t0
            add_log(
                f"CAMOUFLAGE SCAN COMPLETE ({t_camo:.3f}s) — IDENTIFIED {camo_regions_count} POTENTIAL CONCEALMENTS"
            )
        else:
            add_log("CAMOUFLAGE MODULE BYPASSED BY OPERATOR")

        # 4. Anomaly Detection (PatchCore-lite)
        anomaly_map = None
        anomaly_status_label = "BYPASS"
        t_anomaly = 0.0

        if enable_anomaly:
            t0 = time.perf_counter()
            anomaly_status = self.anomaly_detector.get_status()
            anomaly_status_label = anomaly_status["mode"]
            add_log(f"PATCHCORE ANOMALY SCAN [{anomaly_status_label}] IN PROGRESS...")

            anomaly_res = self.anomaly_detector.detect(infer_img)
            anomaly_map = anomaly_res["score_map"]
            t_anomaly = time.perf_counter() - t0
            add_log(
                f"ANOMALY SCAN COMPLETE ({t_anomaly:.3f}s) — PEAK NOVELTY {anomaly_res['max_score']:.2f}"
            )
        else:
            add_log("ANOMALY MODULE BYPASSED BY OPERATOR")

        # 5. Object Detection (YOLOv8)
        detections = []
        t_objects = 0.0

        if enable_objects:
            t0 = time.perf_counter()
            add_log("SCANNING TACTICAL OBJECT TARGETS (YOLOv8)...")
            detections = self.object_detector.detect(infer_img, conf_threshold=object_conf)
            t_objects = time.perf_counter() - t0
            yolo_status = self.object_detector.get_status()
            if yolo_status["online"]:
                add_log(
                    f"OBJECT DETECTION COMPLETE ({t_objects:.3f}s) — RECOGNIZED {len(detections)} CONTACT(S)"
                )
            else:
                add_log(f"OBJECT DETECTION OFFLINE: {yolo_status['error'] or 'Weights unavailable'}")
        else:
            add_log("OBJECT RECOGNITION MODULE BYPASSED BY OPERATOR")

        # 6. Fusion & Suspicion Synthesis
        t0 = time.perf_counter()
        add_log(f"SYNTHESIZING FUSED SUSPICION MAP (CAMO: {camo_weight:.2f}, ANOMALY: {anomaly_weight:.2f})...")

        suspicion_map = self.fusion_engine.compute_suspicion_map(
            camo_map=camo_map,
            anomaly_map=anomaly_map,
            w_camo=camo_weight if enable_camo else 0.0,
            w_anomaly=anomaly_weight if enable_anomaly else 0.0,
            target_shape=(ih, iw),
        )

        # Ranked Regions Extraction (ALPHA .. ECHO)
        ranked_regions = self.fusion_engine.extract_top_regions(
            suspicion_map=suspicion_map,
            threshold=max(0.42, camo_threshold * 0.90),
            max_regions=5,
        )

        # Compute Overall Threat Index (0-100)
        threat_data = self.fusion_engine.compute_threat_index(
            suspicion_map=suspicion_map,
            ranked_regions=ranked_regions,
            detections=detections,
        )

        t_fusion = time.perf_counter() - t0
        add_log(
            f"FUSION COMPLETE ({t_fusion:.3f}s) — THREAT INDEX: {threat_data['index']}/100 [{threat_data['level']}]"
        )

        # 7. Render Tactical Overlays
        t0 = time.perf_counter()
        add_log("RENDERING MULTI-SPECTRAL TACTICAL OVERLAYS...")

        overlay_arrays = self.fusion_engine.render_overlays(
            orig_rgb=infer_img,
            camo_map=camo_map,
            camo_contours=camo_contours,
            anomaly_map=anomaly_map,
            suspicion_map=suspicion_map,
            ranked_regions=ranked_regions,
            detections=detections,
            opacity=overlay_opacity,
        )

        # Encode all overlays to base64 PNG
        encoded_overlays = {
            name: to_base64_png(arr) for name, arr in overlay_arrays.items()
        }

        t_render = time.perf_counter() - t0
        total_time = time.perf_counter() - total_start_time
        add_log(f"MISSION SITREP COMPILED IN {total_time:.3f}s TOTAL")

        # If image was scaled for inference, project region boxes back to original dimensions for export
        export_ranked_regions = []
        inv_scale = 1.0 / scale if scale > 0 else 1.0
        for reg in ranked_regions:
            rx, ry, rw, rh = reg["box"]
            cx, cy = reg["centroid"]
            export_ranked_regions.append(
                {
                    **reg,
                    "box_orig": [
                        int(round(rx * inv_scale)),
                        int(round(ry * inv_scale)),
                        int(round(rw * inv_scale)),
                        int(round(rh * inv_scale)),
                    ],
                    "centroid_orig": [
                        int(round(cx * inv_scale)),
                        int(round(cy * inv_scale)),
                    ],
                }
            )

        export_detections = []
        for det in detections:
            bx1, by1, bx2, by2 = det["box"]
            export_detections.append(
                {
                    **det,
                    "box_orig": [
                        int(round(bx1 * inv_scale)),
                        int(round(by1 * inv_scale)),
                        int(round(bx2 * inv_scale)),
                        int(round(by2 * inv_scale)),
                    ],
                }
            )

        return {
            "success": True,
            "overlays": encoded_overlays,
            "threat_index": threat_data["index"],
            "threat_level": threat_data["level"],
            "threat_color": threat_data["color"],
            "ranked_regions": export_ranked_regions,
            "detections": export_detections,
            "timings": {
                "camo_s": round(t_camo, 3),
                "anomaly_s": round(t_anomaly, 3),
                "objects_s": round(t_objects, 3),
                "fusion_s": round(t_fusion, 3),
                "render_s": round(t_render, 3),
                "total_s": round(total_time, 3),
            },
            "status": {
                "device": self.device_name,
                "camo_engine": self.camo_detector.engine_mode,
                "anomaly_mode": anomaly_status_label,
                "yolo_status": self.object_detector.get_status()["status_label"],
            },
            "dimensions": {
                "original": [orig_w, orig_h],
                "inference": [iw, ih],
                "scale": round(scale, 3),
            },
            "logs": logs,
        }
