"""
TERRASCAN-AI — Tactical Computer Vision Utilities
Image I/O, resizing, color transformations, colormap blending, and base64 serialization.
"""

from __future__ import annotations

import base64
import io
import re
from typing import Tuple, Union, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch


def get_device() -> torch.device:
    """Return the fastest available computing device (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_name() -> str:
    """Return a descriptive label for the active device."""
    if torch.cuda.is_available():
        return f"GPU ({torch.cuda.get_device_name(0)})"
    return "CPU"


def load_image(
    source: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray]
) -> np.ndarray:
    """
    Load an image from various input types and standardize to RGB uint8 ndarray.
    Handles grayscale, RGBA, and palette modes, converting all to standard 3-channel RGB.
    """
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        elif source.ndim == 3:
            if source.shape[2] == 4:
                return cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
            elif source.shape[2] == 3:
                return source.copy()
            elif source.shape[2] == 1:
                return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        raise ValueError(f"Unsupported numpy image shape: {source.shape}")

    if isinstance(source, Image.Image):
        pil_img = source
    elif isinstance(source, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(source))
    elif isinstance(source, io.BytesIO):
        pil_img = Image.open(source)
    elif isinstance(source, str):
        # Could be base64 data URL or file path
        if source.startswith("data:image") or "base64," in source:
            base64_data = re.sub(r"^data:image/.+;base64,", "", source)
            img_bytes = base64.b64decode(base64_data)
            pil_img = Image.open(io.BytesIO(img_bytes))
        else:
            pil_img = Image.open(source)
    else:
        raise TypeError(f"Cannot load image from source of type {type(source)}")

    # Correct EXIF rotation if present
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass

    # Standardize to RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    return np.array(pil_img, dtype=np.uint8)


def resize_image_for_inference(
    img_rgb: np.ndarray,
    max_long_side: int = 1280
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image so its longest edge is at most `max_long_side` while preserving aspect ratio.

    Returns:
        resized_img: np.ndarray (RGB)
        scale: float factor (resized_dim / orig_dim)
        (orig_h, orig_w): original dimensions
    """
    orig_h, orig_w = img_rgb.shape[:2]
    long_side = max(orig_h, orig_w)

    if long_side <= max_long_side:
        return img_rgb.copy(), 1.0, (orig_h, orig_w)

    scale = max_long_side / float(long_side)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale, (orig_h, orig_w)


def normalize_map(score_map: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Min-max normalize a float array to [0.0, 1.0].
    Safely handles uniform arrays without division by zero.
    """
    arr = np.asarray(score_map, dtype=np.float32)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    diff = max_val - min_val

    if diff < eps:
        return np.zeros_like(arr, dtype=np.float32)

    norm = (arr - min_val) / (diff + eps)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def suppress_border_margins(
    score_map: np.ndarray, margin_ratio: float = 0.05
) -> np.ndarray:
    """
    Attenuate artificial convolution boundary spikes around image perimeters
    using a smooth cosine taper and outer frame blanking.
    """
    h, w = score_map.shape[:2]
    my = max(4, int(round(h * margin_ratio)))
    mx = max(4, int(round(w * margin_ratio)))

    mask = np.ones((h, w), dtype=np.float32)

    # Smooth vertical taper
    for y in range(my):
        weight = 0.5 * (1.0 - np.cos(np.pi * y / my))
        mask[y, :] *= weight
        mask[h - 1 - y, :] *= weight

    # Smooth horizontal taper
    for x in range(mx):
        weight = 0.5 * (1.0 - np.cos(np.pi * x / mx))
        mask[:, x] *= weight
        mask[:, w - 1 - x] *= weight

    # Blank out outermost perimeter lines
    mask[:3, :] = 0.0
    mask[-3:, :] = 0.0
    mask[:, :3] = 0.0
    mask[:, -3:] = 0.0

    return (score_map * mask).astype(np.float32)


def robust_saliency_norm(
    score_map: np.ndarray,
    noise_percentile: float = 65.0,
    peak_percentile: float = 99.2,
) -> np.ndarray:
    """
    Statistical contrast-enhancement: suppresses ambient background noise floor
    (below noise_percentile) while cleanly isolating true structural peaks.
    If the image has no significant contrast (e.g. uniform terrain/noise), returns near-zero.
    """
    arr = np.asarray(score_map, dtype=np.float32)
    p_low = float(np.percentile(arr, noise_percentile))
    p_high = float(np.percentile(arr, peak_percentile))
    span = p_high - p_low

    # Contrast significance gate: if dynamic range is negligible, suppress as uniform background
    if span < 1e-4:
        return np.zeros_like(arr, dtype=np.float32)

    # Shift so background floor sits at zero
    shifted = np.maximum(0.0, arr - p_low)
    norm = np.clip(shifted / span, 0.0, 1.0)
    # Power curve to sharply attenuate sub-threshold background variations
    enhanced = np.power(norm, 1.4)
    return enhanced.astype(np.float32)


def to_base64_png(image: Union[np.ndarray, Image.Image], add_header: bool = True) -> str:
    """
    Encode an RGB uint8 image array or PIL image to a base64 PNG data string.
    """
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        raise TypeError(f"Unsupported image type for base64: {type(image)}")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG", optimize=True)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    if add_header:
        return f"data:image/png;base64,{b64_str}"
    return b64_str


def apply_colormap(
    score_map: np.ndarray,
    colormap_name: str = "jet"
) -> np.ndarray:
    """
    Convert a [0, 1] float map to an RGB heatmap using OpenCV colormaps.
    Default: cv2.COLORMAP_JET.
    """
    norm = np.clip(score_map, 0.0, 1.0)
    uint8_map = (norm * 255.0).astype(np.uint8)

    cmap_code = cv2.COLORMAP_JET
    if colormap_name.lower() == "turbo":
        cmap_code = cv2.COLORMAP_TURBO
    elif colormap_name.lower() == "viridis":
        cmap_code = cv2.COLORMAP_VIRIDIS
    elif colormap_name.lower() == "hot":
        cmap_code = cv2.COLORMAP_HOT

    bgr_heatmap = cv2.applyColorMap(uint8_map, cmap_code)
    rgb_heatmap = cv2.cvtColor(bgr_heatmap, cv2.COLOR_BGR2RGB)
    return rgb_heatmap


def blend_overlay(
    base_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    alpha: float = 0.55
) -> np.ndarray:
    """
    Blend base RGB image with an overlay RGB image:
    blended = (1 - alpha) * base + alpha * overlay
    """
    base_f = base_rgb.astype(np.float32)
    overlay_f = overlay_rgb.astype(np.float32)
    alpha = np.clip(alpha, 0.0, 1.0)

    blended = (1.0 - alpha) * base_f + alpha * overlay_f
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def draw_tactical_box(
    canvas: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    score: float,
    color: Tuple[int, int, int] = (154, 205, 50)  # tactical olive/lime
) -> None:
    """
    Draw a tactical military-style bounding box with corner reticles and badge.
    box: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = canvas.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    # Main rectangle
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Tactical corner brackets (L-shape accents)
    arm_len = min(16, max(6, (x2 - x1) // 5), max(6, (y2 - y1) // 5))
    thick = 2
    # Top-Left
    cv2.line(canvas, (x1, y1), (x1 + arm_len, y1), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x1, y1), (x1, y1 + arm_len), color, thick, cv2.LINE_AA)
    # Top-Right
    cv2.line(canvas, (x2, y1), (x2 - arm_len, y1), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1), (x2, y1 + arm_len), color, thick, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(canvas, (x1, y2), (x1 + arm_len, y2), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x1, y2), (x1, y2 - arm_len), color, thick, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(canvas, (x2, y2), (x2 - arm_len, y2), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x2, y2), (x2, y2 - arm_len), color, thick, cv2.LINE_AA)

    # Label badge with dark background
    badge_text = f"{label.upper()} {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    font_thick = 1
    (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, font_thick)

    tag_y1 = max(0, y1 - th - 6)
    tag_y2 = y1
    tag_x2 = min(w, x1 + tw + 8)

    # Dark background for tactical readability
    cv2.rectangle(canvas, (x1, tag_y1), (tag_x2, tag_y2), (16, 21, 12), -1)
    cv2.rectangle(canvas, (x1, tag_y1), (tag_x2, tag_y2), color, 1)
    cv2.putText(
        canvas,
        badge_text,
        (x1 + 4, y1 - 4),
        font,
        font_scale,
        color,
        font_thick,
        cv2.LINE_AA,
    )


def draw_tactical_marker(
    canvas: np.ndarray,
    center: Tuple[int, int],
    rank_label: str,
    score: float,
    color: Tuple[int, int, int] = (214, 69, 69)  # red alert accent
) -> None:
    """
    Draw a tactical target crosshair marker with military designation (ALPHA, BRAVO, etc.).
    """
    cx, cy = int(center[0]), int(center[1])
    h, w = canvas.shape[:2]
    cx, cy = max(10, min(w - 10, cx)), max(10, min(h - 10, cy))

    radius = 14
    # Concentric rings
    cv2.circle(canvas, (cx, cy), radius, color, 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 2, color, -1, cv2.LINE_AA)

    # Crosshair reticles
    gap = 4
    reticle_len = 10
    # Left, Right, Top, Bottom
    cv2.line(canvas, (cx - radius - reticle_len, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx + gap, cy), (cx + radius + reticle_len, cy), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - radius - reticle_len), (cx, cy - gap), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy + gap), (cx, cy + radius + reticle_len), color, 1, cv2.LINE_AA)

    # Callout badge
    badge_text = f"[{rank_label}] {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, 1)

    bx1 = min(w - tw - 8, cx + radius + 4)
    by1 = max(2, cy - th // 2 - 3)
    bx2 = bx1 + tw + 6
    by2 = by1 + th + 6

    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (16, 21, 12), -1)
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 1)
    cv2.putText(
        canvas,
        badge_text,
        (bx1 + 3, by1 + th + 1),
        font,
        font_scale,
        color,
        1,
        cv2.LINE_AA,
    )
