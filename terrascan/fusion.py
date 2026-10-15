"""
TERRASCAN-AI — Fusion and Multi-Sensor Synthesis Engine
Fuses camouflage, anomaly, and object signals into a unified tactical suspicion map,
ranks key regions (ALPHA through ECHO), computes the 0-100 Threat Index,
and renders multi-layer tactical visualization overlays with intelligent false-positive suppression.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np

from terrascan.utils import (
    normalize_map,
    apply_colormap,
    blend_overlay,
    draw_tactical_box,
    draw_tactical_marker,
    suppress_border_margins,
    robust_saliency_norm,
)

RANK_DESIGNATIONS = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO"]


class FusionEngine:
    """
    Synthesizes camouflage, anomaly, and object signals into actionable tactical intel.
    Filters out background clutter, frame boundary reflections, and giant full-screen masks.
    """

    def __init__(self):
        pass

    def compute_suspicion_map(
        self,
        camo_map: Optional[np.ndarray],
        anomaly_map: Optional[np.ndarray],
        w_camo: float = 0.6,
        w_anomaly: float = 0.4,
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Synthesize fused suspicion map:
        suspicion = (w_camo * camo + w_anomaly * anomaly) + agreement_bonus
        """
        if target_shape is not None:
            h, w = target_shape
        elif camo_map is not None:
            h, w = camo_map.shape[:2]
        elif anomaly_map is not None:
            h, w = anomaly_map.shape[:2]
        else:
            raise ValueError("At least one map or target_shape must be provided.")

        camo = (
            camo_map
            if camo_map is not None
            else np.zeros((h, w), dtype=np.float32)
        )
        anomaly = (
            anomaly_map
            if anomaly_map is not None
            else np.zeros((h, w), dtype=np.float32)
        )

        total_w = w_camo + w_anomaly
        if total_w > 0:
            wc = w_camo / total_w
            wa = w_anomaly / total_w
        else:
            wc, wa = 0.6, 0.4

        # Agreement bonus: where both detectors identify high activity
        agreement = camo * anomaly
        agreement_bonus = 0.25 * agreement

        raw_suspicion = (wc * camo) + (wa * anomaly) + agreement_bonus
        # Suppress frame perimeter artifacts
        suppressed = suppress_border_margins(raw_suspicion, margin_ratio=0.025)
        # Apply statistical contrast enhancement
        fused_norm = robust_saliency_norm(suppressed, noise_percentile=60.0, peak_percentile=99.2)
        return fused_norm

    def extract_top_regions(
        self,
        suspicion_map: np.ndarray,
        threshold: float = 0.48,
        max_regions: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Extract connected components above threshold and rank by tactical priority.
        Rejects full-screen background blobs, frame edge slivers, and low-confidence noise.
        """
        h, w = suspicion_map.shape[:2]
        total_pixels = float(h * w)

        # Binary thresholding
        thresh_val = float(np.clip(threshold, 0.35, 0.85))
        binary = (suspicion_map >= thresh_val).astype(np.uint8) * 255

        # Morphological opening to detach thin background bridges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Connected components with statistics
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )

        candidates = []
        min_area = total_pixels * 0.0025  # Minimum 0.25% of image area (~1,000 px on 640x640)
        max_area = total_pixels * 0.30    # Maximum 30% of image area (rejects full canopy covers)

        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue

            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            bw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[label_idx]

            # Reject long thin strips (non-compact noise) and boundary ringing
            aspect_ratio = max(bw, bh) / max(1.0, float(min(bw, bh)))
            if aspect_ratio > 4.0:
                continue

            touches_border = (x <= 8 or y <= 8 or x + bw >= w - 8 or y + bh >= h - 8)
            if touches_border and (bw < 32 or bh < 32 or aspect_ratio > 2.5):
                continue
            if bw > 0.80 * w and bh > 0.80 * h:
                continue

            mask = (labels == label_idx)
            region_scores = suspicion_map[mask]
            if len(region_scores) == 0:
                continue

            mean_score = float(np.mean(region_scores))
            max_score = float(np.max(region_scores))

            # Must meet minimum confidence criteria to be classified as a real threat
            if mean_score < 0.45 or max_score < 0.58:
                continue

            area_pct = round((area / total_pixels) * 100.0, 2)
            priority = (mean_score * 0.65) + (max_score * 0.35) + (min(area_pct, 5.0) * 0.01)

            candidates.append(
                {
                    "box": [x, y, bw, bh],
                    "box_xyxy": [x, y, x + bw, y + bh],
                    "centroid": [int(round(cx)), int(round(cy))],
                    "area_pct": area_pct,
                    "mean_score": round(mean_score, 3),
                    "max_score": round(max_score, 3),
                    "priority": priority,
                }
            )

        # Sort descending by priority
        candidates.sort(key=lambda item: item["priority"], reverse=True)
        top_candidates = candidates[:max_regions]

        # Assign designations (ALPHA, BRAVO...)
        ranked_regions = []
        for idx, reg in enumerate(top_candidates):
            rank_name = (
                RANK_DESIGNATIONS[idx]
                if idx < len(RANK_DESIGNATIONS)
                else f"SECTOR-{idx+1}"
            )
            reg["rank"] = rank_name
            ranked_regions.append(reg)

        return ranked_regions

    def compute_threat_index(
        self,
        suspicion_map: np.ndarray,
        ranked_regions: List[Dict[str, Any]],
        detections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute tactical Threat Index (0-100).
        If image is nominal background terrain, Threat Index stays LOW (0-20).
        Only escalates when significant regions or objects are validated.
        """
        if suspicion_map.size == 0:
            return {"index": 0, "level": "LOW / CLEAR", "color": "#3d4a2a"}

        p98 = float(np.percentile(suspicion_map, 98))

        # If no significant target regions and no detected objects exist:
        if not ranked_regions and not detections:
            index_val = int(np.clip(p98 * 25.0, 0, 24))
            return {
                "index": index_val,
                "level": "LOW / CLEAR",
                "color": "#3d4a2a",
                "p98_suspicion": round(p98, 3),
            }

        # Regional threat contribution
        if ranked_regions:
            top_mean = float(np.mean([r["mean_score"] for r in ranked_regions[:2]]))
            top_max = float(max([r["max_score"] for r in ranked_regions]))
            area_sum = float(sum([r["area_pct"] for r in ranked_regions[:2]]))
            base_score = (top_mean * 50.0) + (top_max * 25.0) + min(15.0, area_sum * 1.5)
        else:
            base_score = p98 * 40.0

        # Object boost
        object_boost = 0.0
        if detections:
            max_conf = max([d["score"] for d in detections])
            object_boost = min(15.0, max_conf * 15.0)

        raw_index = base_score + object_boost
        threat_index = int(np.clip(round(raw_index), 0, 100))

        # Threat classification
        if threat_index >= 75:
            level = "CRITICAL ALERT"
            color = "#d64545"  # red
        elif threat_index >= 50:
            level = "HIGH CONCERN"
            color = "#e0a526"  # amber
        elif threat_index >= 25:
            level = "ELEVATED"
            color = "#9acd32"  # tactical green
        else:
            level = "LOW / CLEAR"
            color = "#3d4a2a"  # subdued olive

        return {
            "index": threat_index,
            "level": level,
            "color": color,
            "p98_suspicion": round(p98, 3),
        }

    def render_overlays(
        self,
        orig_rgb: np.ndarray,
        camo_map: Optional[np.ndarray],
        camo_contours: List[Any],
        anomaly_map: Optional[np.ndarray],
        suspicion_map: np.ndarray,
        ranked_regions: List[Dict[str, Any]],
        detections: List[Dict[str, Any]],
        opacity: float = 0.60,
    ) -> Dict[str, np.ndarray]:
        """
        Generate clean tactical visualization layers without noisy clutter.
        """
        h, w = orig_rgb.shape[:2]

        # 1. Original
        view_original = orig_rgb.copy()

        # 2. Camouflage overlay
        view_camo = orig_rgb.copy()
        if camo_map is not None:
            lime_color = np.zeros_like(orig_rgb)
            lime_color[:, :] = [154, 205, 50]
            alpha_mask = np.expand_dims(camo_map * opacity, axis=-1)
            view_camo = (
                (1.0 - alpha_mask) * view_camo.astype(np.float32)
                + alpha_mask * lime_color.astype(np.float32)
            ).astype(np.uint8)

        # Draw contour boundaries only for filtered regions (max 3)
        if camo_contours:
            cv2.drawContours(view_camo, camo_contours[:3], -1, (154, 205, 50), 2, cv2.LINE_AA)

        # 3. Anomaly Heatmap (JET colormap)
        view_anomaly = orig_rgb.copy()
        if anomaly_map is not None:
            anomaly_colored = apply_colormap(anomaly_map, "jet")
            view_anomaly = blend_overlay(orig_rgb, anomaly_colored, alpha=opacity)

        # 4. Objects Detection View
        view_objects = orig_rgb.copy()
        for det in detections:
            box = det["box"]
            label = det["label"]
            score = det["score"]
            draw_tactical_box(view_objects, box, label, score, color=(154, 205, 50))

        # 5. Suspicion Heatmap
        suspicion_colored = apply_colormap(suspicion_map, "jet")
        view_suspicion = blend_overlay(orig_rgb, suspicion_colored, alpha=opacity)

        # 6. Combined "All Layers" Tactical View
        view_all = blend_overlay(orig_rgb, suspicion_colored, alpha=max(0.30, opacity * 0.70))

        if camo_contours:
            cv2.drawContours(view_all, camo_contours[:3], -1, (154, 205, 50), 2, cv2.LINE_AA)

        for det in detections:
            box = det["box"]
            label = det["label"]
            score = det["score"]
            draw_tactical_box(view_all, box, label, score, color=(224, 165, 38))

        # Draw verified ranked regions
        for reg in ranked_regions:
            center = reg["centroid"]
            rank = reg["rank"]
            score = reg["mean_score"]
            draw_tactical_marker(view_all, center, rank, score, color=(214, 69, 69))

            rx, ry, rw, rh = reg["box"]
            cv2.rectangle(view_all, (rx, ry), (rx + rw, ry + rh), (214, 69, 69), 1, cv2.LINE_AA)

        return {
            "original": view_original,
            "camouflage": view_camo,
            "anomaly": view_anomaly,
            "objects": view_objects,
            "suspicion": view_suspicion,
            "all_layers": view_all,
        }
