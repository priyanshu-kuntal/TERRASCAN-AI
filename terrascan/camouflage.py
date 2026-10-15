"""
TERRASCAN-AI — Camouflage Detection Engine
Dual-mode camouflage detector:
- Mode A: Supervised COD (Camouflaged Object Detection) network when weights exist in models/
- Mode B: Unsupervised frequency-domain spectral residual saliency + deep patch clustering
          with background noise floor suppression and border margin artifact removal.
"""

from __future__ import annotations

import os
import glob
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch
import torch.nn as nn
import torch.nn.functional as F

from terrascan.utils import (
    get_device,
    normalize_map,
    suppress_border_margins,
    robust_saliency_norm,
)


def compute_spectral_residual_saliency(img_rgb: np.ndarray) -> np.ndarray:
    """
    Computes Hou & Zhang spectral residual visual saliency in the Fourier domain.
    Mathematically cancels out repeating natural textures (dense foliage, grass, waves)
    by subtracting the average log amplitude spectrum, highlighting true concealed objects.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape[:2]
    opt_h = cv2.getOptimalDFTSize(h)
    opt_w = cv2.getOptimalDFTSize(w)
    padded = cv2.copyMakeBorder(gray, 0, opt_h - h, 0, opt_w - w, cv2.BORDER_REFLECT)

    dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    magnitude, angle = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])

    # Log amplitude spectrum & locally smoothed background spectrum
    log_amplitude = np.log(magnitude + 1e-8)
    avg_log_amplitude = cv2.boxFilter(log_amplitude, -1, (5, 5))

    # Spectral residual
    spectral_residual = log_amplitude - avg_log_amplitude

    # Reconstruct complex spectrum with original phase
    re_x, re_y = cv2.polarToCart(np.exp(spectral_residual), angle)
    res_dft = np.stack([re_x, re_y], axis=-1)

    # Inverse DFT back to spatial domain
    idft = cv2.idft(res_dft)
    sal = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])[:h, :w]
    sal_sq = sal ** 2
    sal_smooth = cv2.GaussianBlur(sal_sq, (17, 17), 4.0)
    return sal_smooth


class CODNetwork(nn.Module):
    """
    PraNet / SINet style Camouflaged Object Detection segmentation head
    used when a trained checkpoint is provided in models/.
    """
    def __init__(self, in_channels: int = 512, hidden_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels // 2)
        self.out_conv = nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        feat = self.relu(self.bn1(self.conv1(x)))
        feat = self.relu(self.bn2(self.conv2(feat)))
        logits = self.out_conv(feat)
        prob = torch.sigmoid(F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False))
        return prob


class CamouflageDetector:
    """
    Camouflage Detector supporting both COD-NET (supervised) and UNSUPERVISED modes.
    """

    def __init__(self, models_dir: str = "models", device: Optional[torch.device] = None):
        self.models_dir = models_dir
        self.device = device or get_device()
        self.engine_mode: str = "UNSUPERVISED"
        self.cod_model: Optional[nn.Module] = None
        self.backbone: Optional[nn.Module] = None

        # Scan models/ for trained COD checkpoints
        self._init_cod_model()

        # If COD not loaded, initialize backbone for Mode B
        if self.cod_model is None:
            self._init_backbone()

    def _init_cod_model(self) -> None:
        """Scan models/ for trained COD checkpoints."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            return

        checkpoint_patterns = [
            os.path.join(self.models_dir, "*cod*.pth"),
            os.path.join(self.models_dir, "*camo*.pth"),
            os.path.join(self.models_dir, "*sinet*.pth"),
            os.path.join(self.models_dir, "*pranet*.pth"),
            os.path.join(self.models_dir, "*.pth"),
        ]

        found_checkpoints = []
        for pat in checkpoint_patterns:
            found_checkpoints.extend(glob.glob(pat))

        if found_checkpoints:
            ckpt_path = found_checkpoints[0]
            try:
                model = CODNetwork()
                state = torch.load(ckpt_path, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)
                model.to(self.device).eval()
                self.cod_model = model
                self.engine_mode = "COD-NET"
            except Exception:
                self.cod_model = None
                self.engine_mode = "UNSUPERVISED"
        else:
            self.engine_mode = "UNSUPERVISED"

    def _init_backbone(self) -> None:
        """Initialize torchvision backbone for dense patch representations."""
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            try:
                weights = ResNet18_Weights.DEFAULT
                net = resnet18(weights=weights)
            except Exception:
                net = resnet18(weights=None)

            modules = list(net.children())[:-3]
            self.backbone = nn.Sequential(*modules).to(self.device).eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
        except Exception:
            self.backbone = None

    def _extract_deep_features(self, img_rgb: np.ndarray) -> np.ndarray:
        """Extract multi-scale patch feature map from PyTorch backbone."""
        h, w = img_rgb.shape[:2]
        if self.backbone is None:
            return self._extract_fallback_features(img_rgb)

        try:
            in_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            in_tensor = (in_tensor - mean) / std
            in_tensor = in_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                feats = self.backbone(in_tensor)
                feats = F.normalize(feats, p=2, dim=1)
                feat_np = feats.squeeze(0).permute(1, 2, 0).cpu().numpy()
                return feat_np
        except Exception:
            return self._extract_fallback_features(img_rgb)

    def _extract_fallback_features(self, img_rgb: np.ndarray) -> np.ndarray:
        """Handcrafted multi-scale Gabor and texture feature descriptors."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        features = [cv2.resize(gray, (gray.shape[1] // 8, gray.shape[0] // 8))]

        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            kernel = cv2.getGaborKernel((9, 9), 3.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            fimg = cv2.filter2D(gray, cv2.CV_32F, kernel)
            features.append(cv2.resize(fimg, (gray.shape[1] // 8, gray.shape[0] // 8)))

        stacked = np.stack(features, axis=-1)
        return stacked

    def _unsupervised_camo_estimation(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Unsupervised camouflage estimation combining:
        1. Spectral residual frequency-domain visual saliency
        2. Deep patch feature clustering (K-Means, k=4)
        3. CIE-LAB color disparity against ambient foliage surround
        4. Statistical background noise floor suppression
        """
        h, w = img_rgb.shape[:2]

        # --- 1. Spectral Residual Saliency (suppresses homogeneous foliage repetition) ---
        spectral_sal = compute_spectral_residual_saliency(img_rgb)
        spectral_norm = normalize_map(spectral_sal)

        # --- 2. Deep Feature Clustering ---
        feat_map = self._extract_deep_features(img_rgb)
        fh, fw, c = feat_map.shape
        flat_feats = feat_map.reshape(-1, c)

        k = min(4, max(2, len(flat_feats) // 50))
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256, n_init=3)
        cluster_labels = kmeans.fit_predict(flat_feats)

        counts = np.bincount(cluster_labels, minlength=k)
        dominant_cluster = int(np.argmax(counts))
        dominant_center = kmeans.cluster_centers_[dominant_cluster]

        # Distance to dominant background center
        dists = np.linalg.norm(flat_feats - dominant_center, axis=1)
        cluster_scores = dists.reshape(fh, fw)
        cluster_scores = cv2.resize(cluster_scores, (w, h), interpolation=cv2.INTER_CUBIC)
        cluster_norm = normalize_map(cluster_scores)

        # --- 3. CIE-LAB Color Disparity vs Surrounding Environment ---
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        ksize = max(25, (min(h, w) // 12) | 1)
        lab_surround = cv2.GaussianBlur(lab, (ksize, ksize), sigmaX=ksize / 3.0)
        lab_diff = np.sqrt(np.sum((lab - lab_surround) ** 2, axis=2))
        lab_norm = normalize_map(lab_diff)

        # --- Fused Synthesis ---
        fused = (
            0.50 * spectral_norm +
            0.35 * cluster_norm +
            0.15 * lab_norm
        )
        fused = cv2.GaussianBlur(fused, (11, 11), 3.0)
        fused = suppress_border_margins(fused, margin_ratio=0.03)

        # Statistical noise suppression: cuts off the lowest 60% ambient noise
        clean_map = robust_saliency_norm(fused, noise_percentile=60.0, peak_percentile=99.0)
        return clean_map

    def detect(self, img_rgb: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
        """
        Execute camouflage detection on an RGB image.
        """
        h, w = img_rgb.shape[:2]
        total_pixels = float(h * w)

        if self.cod_model is not None and self.engine_mode == "COD-NET":
            try:
                in_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                in_tensor = in_tensor.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    prob = self.cod_model(in_tensor, (h, w))
                    score_map = prob.squeeze().cpu().numpy().astype(np.float32)
                    score_map = normalize_map(score_map)
            except Exception:
                score_map = self._unsupervised_camo_estimation(img_rgb)
        else:
            score_map = self._unsupervised_camo_estimation(img_rgb)

        score_map = normalize_map(score_map)

        # Dynamic threshold based on user input (scaled appropriately)
        thresh_val = float(np.clip(threshold * 0.9, 0.20, 0.85))
        binary = (score_map >= thresh_val).astype(np.uint8) * 255

        # Morphological clean up
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

        # Connected components and contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out tiny insignificant specks (< 0.2% area) AND whole-screen blobs (> 35% area)
        min_area = total_pixels * 0.002
        max_area = total_pixels * 0.35
        filtered_contours = []

        for c in contours:
            c_area = cv2.contourArea(c)
            if min_area <= c_area <= max_area:
                bx, by, bw, bh = cv2.boundingRect(c)
                # Ignore boundary slivers
                if (bx <= 2 or by <= 2 or bx + bw >= w - 2 or by + bh >= h - 2) and (bw < 20 or bh < 20):
                    continue
                filtered_contours.append(c)

        # Sort descending by area and take at most top 5 true target contours
        filtered_contours.sort(key=cv2.contourArea, reverse=True)
        filtered_contours = filtered_contours[:5]

        # Convert contours to serializable coordinate lists
        serializable_contours = []
        for c in filtered_contours:
            pts = c.squeeze().tolist()
            if isinstance(pts, list) and len(pts) > 0:
                if isinstance(pts[0], int):
                    pts = [pts]
                serializable_contours.append(pts)

        return {
            "score_map": score_map,
            "binary_mask": cleaned,
            "contours": filtered_contours,
            "serializable_contours": serializable_contours,
            "regions_count": len(filtered_contours),
            "engine": self.engine_mode,
        }
