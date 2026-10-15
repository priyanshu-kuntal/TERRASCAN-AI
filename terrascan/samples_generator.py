"""
TERRASCAN-AI — Synthetic Tactical Terrain Sample Generator
Generates realistic procedural terrain imagery for testing without external downloads:
1. sample_01.png: Dense jungle foliage with low-contrast camouflaged bunker/cache
2. sample_02.png: Aerial forest canopy with geometric thermal/visual anomaly
3. sample_03.png: Scrubland with concealed tactical vehicle silhouette & netting
"""

import os
import cv2
import numpy as np


def generate_cellular_noise(h: int, w: int, scale: float = 20.0) -> np.ndarray:
    """Generate multi-scale cellular noise for terrain synthesis."""
    coords_y, coords_x = np.mgrid[0:h, 0:w]
    noise = np.zeros((h, w), dtype=np.float32)

    # Multi-frequency octaves
    for octave in [1.0, 2.0, 4.0, 8.0]:
        sub_h = max(4, int(h / (scale / octave)))
        sub_w = max(4, int(w / (scale / octave)))
        rand_grid = np.random.uniform(0.0, 1.0, (sub_h, sub_w)).astype(np.float32)
        upscaled = cv2.resize(rand_grid, (w, h), interpolation=cv2.INTER_CUBIC)
        noise += upscaled * (1.0 / octave)

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    return noise


def generate_sample_01(h: int = 640, w: int = 640) -> np.ndarray:
    """Sample 01: Dense foliage with camouflaged bunker/tent cache."""
    # Base foliage green/olive gradient
    foliage_noise = generate_cellular_noise(h, w, scale=35.0)
    detail_noise = generate_cellular_noise(h, w, scale=12.0)

    # Forest color palette (dark green to olive-brown)
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for c, (low, high) in enumerate([(25, 65), (55, 115), (20, 50)]):  # R, G, B
        channel = low + (high - low) * (0.7 * foliage_noise + 0.3 * detail_noise)
        bg[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    # Sprinkle leaf/branch micro-textures
    speckles = np.random.poisson(lam=12, size=(h, w))
    bg[:, :, 1] = np.clip(bg[:, :, 1] + speckles * 2, 0, 255).astype(np.uint8)

    # Embedded low-contrast camouflaged bunker
    cx, cy = int(w * 0.48), int(h * 0.52)
    rx, ry = int(w * 0.16), int(h * 0.12)

    # Elliptical mask with feathered edges
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (rx, ry), angle=25, startAngle=0, endAngle=360, color=1.0, thickness=-1)
    mask = cv2.GaussianBlur(mask, (31, 31), 9.0)

    # Camouflage pattern inside bunker: subtle disruption pattern (slight color shift & orientation)
    camo_patch = bg.copy()
    camo_noise = generate_cellular_noise(h, w, scale=8.0)
    camo_patch[:, :, 0] = np.clip(camo_patch[:, :, 0] + (camo_noise * 30 - 15), 0, 255)
    camo_patch[:, :, 1] = np.clip(camo_patch[:, :, 1] - (camo_noise * 18 - 8), 0, 255)
    camo_patch[:, :, 2] = np.clip(camo_patch[:, :, 2] + (camo_noise * 22 - 10), 0, 255)

    # Structural seam/netting lines on bunker
    for offset in range(-rx, rx, 18):
        pt1 = (cx + offset - 10, cy - ry + 10)
        pt2 = (cx + offset + 20, cy + ry - 10)
        cv2.line(camo_patch, pt1, pt2, (38, 70, 30), 1, cv2.LINE_AA)

    # Alpha blend bunker into foliage
    mask_3d = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
    blended = (1.0 - mask_3d) * bg.astype(np.float32) + mask_3d * camo_patch.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def generate_sample_02(h: int = 640, w: int = 640) -> np.ndarray:
    """Sample 02: Aerial forest canopy with geometric thermal/visual anomaly."""
    # Aerial tree crown canopy texture
    canopy_noise = generate_cellular_noise(h, w, scale=45.0)
    texture_noise = generate_cellular_noise(h, w, scale=15.0)

    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for c, (low, high) in enumerate([(35, 75), (70, 130), (25, 60)]):
        channel = low + (high - low) * (0.6 * canopy_noise + 0.4 * texture_noise)
        bg[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    # Add tree crown shadowing
    shadows = (canopy_noise < 0.35).astype(np.float32)
    shadows = cv2.GaussianBlur(shadows, (15, 15), 4.0)
    bg = (bg.astype(np.float32) * (1.0 - 0.3 * np.expand_dims(shadows, -1))).astype(np.uint8)

    # Anomaly patch: rectangular clearing / non-natural rectangular installation
    ax1, ay1 = int(w * 0.55), int(h * 0.35)
    ax2, ay2 = ax1 + int(w * 0.22), ay1 + int(h * 0.18)

    anomaly_mask = np.zeros((h, w), dtype=np.float32)
    cv2.rectangle(anomaly_mask, (ax1, ay1), (ax2, ay2), 1.0, -1)
    anomaly_mask = cv2.GaussianBlur(anomaly_mask, (15, 15), 3.0)

    # Anomaly appearance: metallic / concrete tarp with distinct reflectance
    anomaly_fill = np.zeros_like(bg)
    anomaly_fill[:, :] = [95, 110, 88]  # grayish-olive tarp
    grid_noise = generate_cellular_noise(h, w, scale=6.0)
    anomaly_fill = (anomaly_fill.astype(np.float32) + (grid_noise[:, :, None] * 40 - 20))
    anomaly_fill = np.clip(anomaly_fill, 0, 255).astype(np.uint8)

    # Internal geometric layout
    cv2.rectangle(anomaly_fill, (ax1 + 8, ay1 + 8), (ax2 - 8, ay2 - 8), (70, 85, 65), 2)
    cv2.line(anomaly_fill, (ax1 + 8, ay1 + 8), (ax2 - 8, ay2 - 8), (75, 90, 70), 1)

    mask_3d = np.repeat(np.expand_dims(anomaly_mask, axis=-1), 3, axis=-1)
    fused = (1.0 - mask_3d) * bg.astype(np.float32) + mask_3d * anomaly_fill.astype(np.float32)
    return np.clip(fused, 0, 255).astype(np.uint8)


def generate_sample_03(h: int = 640, w: int = 640) -> np.ndarray:
    """Sample 03: Arid scrubland with concealed tactical vehicle silhouette & netting."""
    # Arid ground (brownish-tan, dry shrubs, gravel)
    terrain_noise = generate_cellular_noise(h, w, scale=30.0)
    gravel_noise = generate_cellular_noise(h, w, scale=8.0)

    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for c, (low, high) in enumerate([(85, 140), (80, 125), (55, 90)]):  # tan/dry scrub
        channel = low + (high - low) * (0.65 * terrain_noise + 0.35 * gravel_noise)
        bg[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    # Sparse scrub bushes (dark olive spots)
    num_bushes = 35
    np.random.seed(101)
    for _ in range(num_bushes):
        bx = np.random.randint(20, w - 20)
        by = np.random.randint(20, h - 20)
        br = np.random.randint(12, 35)
        cv2.circle(bg, (bx, by), br, (45, 65, 35), -1)

    bg = cv2.GaussianBlur(bg, (5, 5), 1.0)

    # Hidden vehicle silhouette in center-left
    vx, vy = int(w * 0.35), int(h * 0.55)
    vw, vh = int(w * 0.28), int(h * 0.16)

    vehicle_canvas = bg.copy()
    # Chassis outline
    cv2.rectangle(vehicle_canvas, (vx, vy), (vx + vw, vy + vh), (50, 60, 42), -1)
    # Cab roof
    cv2.rectangle(vehicle_canvas, (vx + int(vw * 0.2), vy - int(vh * 0.4)), (vx + int(vw * 0.75), vy), (55, 65, 45), -1)

    # Disruptive camouflage netting pattern over vehicle
    camo_net = generate_cellular_noise(h, w, scale=7.0)
    net_mask = (camo_net > 0.55).astype(np.float32)
    vehicle_canvas[vy - int(vh * 0.4):vy + vh, vx:vx + vw, 0] = np.clip(
        vehicle_canvas[vy - int(vh * 0.4):vy + vh, vx:vx + vw, 0] + 25, 0, 255
    )
    vehicle_canvas[vy - int(vh * 0.4):vy + vh, vx:vx + vw, 1] = np.clip(
        vehicle_canvas[vy - int(vh * 0.4):vy + vh, vx:vx + vw, 1] + 15, 0, 255
    )

    # Soft feather boundary for netting
    vmask = np.zeros((h, w), dtype=np.float32)
    cv2.rectangle(vmask, (vx - 6, vy - int(vh * 0.45)), (vx + vw + 6, vy + vh + 6), 1.0, -1)
    vmask = cv2.GaussianBlur(vmask, (25, 25), 7.0)

    mask_3d = np.repeat(np.expand_dims(vmask, axis=-1), 3, axis=-1)
    fused = (1.0 - mask_3d) * bg.astype(np.float32) + mask_3d * vehicle_canvas.astype(np.float32)
    return np.clip(fused, 0, 255).astype(np.uint8)


def ensure_sample_images(samples_dir: str = "samples") -> list[str]:
    """Ensure that the 3 synthetic sample images are created on disk."""
    os.makedirs(samples_dir, exist_ok=True)
    sample_files = [
        ("sample_01.png", generate_sample_01),
        ("sample_02.png", generate_sample_02),
        ("sample_03.png", generate_sample_03),
    ]

    paths = []
    for filename, gen_fn in sample_files:
        filepath = os.path.join(samples_dir, filename)
        if not os.path.exists(filepath):
            img_rgb = gen_fn()
            # Save as BGR for OpenCV
            cv2.imwrite(filepath, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        paths.append(filepath)

    return paths


if __name__ == "__main__":
    generated = ensure_sample_images()
    print(f"Generated samples: {generated}")
