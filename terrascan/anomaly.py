"""
TERRASCAN-AI — Anomaly Detection Engine
PatchCore-lite anomaly detector with dual operational modes:
- Reference Mode: Uses terrain images in data/reference/ to build a kNN patch memory bank.
- Self-Reference Mode: Scores visual outliers against the image's own internal terrain statistics.
"""

from __future__ import annotations

import os
import glob
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F

from terrascan.utils import get_device, normalize_map, load_image


class PatchCoreFeatureExtractor(nn.Module):
    """
    PatchCore-lite feature extractor utilizing WideResNet50 (layer2 + layer3)
    with adaptive spatial neighborhood pooling.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self._init_network()

    def _init_network(self) -> None:
        try:
            from torchvision.models import wide_resnet50_2, WideResNet50_2_Weights
            try:
                weights = WideResNet50_2_Weights.DEFAULT
                net = wide_resnet50_2(weights=weights)
            except Exception:
                net = wide_resnet50_2(weights=None)
        except Exception:
            # Fallback to ResNet50 or ResNet18 if WideResNet not available
            from torchvision.models import resnet18, ResNet18_Weights
            try:
                weights = ResNet18_Weights.DEFAULT
                net = resnet18(weights=weights)
            except Exception:
                net = resnet18(weights=None)

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

        self.to(self.device).eval()
        for p in self.parameters():
            p.requires_grad = False

        # Neighborhood pooling kernel
        self.patch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def extract_patches(self, x: torch.Tensor) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract locally aggregated patch features from layer2 and layer3.
        Returns:
            flat_patches: np.ndarray of shape (N, C)
            (feat_h, feat_w): spatial resolution of patch grid
        """
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            feat_layer2 = self.layer2(x)
            feat_layer3 = self.layer3(feat_layer2)

            # Spatial pooling to capture local patch context
            f2 = self.patch_pool(feat_layer2)
            f3 = self.patch_pool(feat_layer3)

            # Align layer3 spatial resolution to layer2
            f3_aligned = F.interpolate(
                f3, size=(f2.shape[2], f2.shape[3]), mode="bilinear", align_corners=False
            )

            # Concatenate along channel dimension
            f_cat = torch.cat([f2, f3_aligned], dim=1)  # (1, C, fh, fw)
            f_norm = F.normalize(f_cat, p=2, dim=1)

            feat_h, feat_w = f_norm.shape[2], f_norm.shape[3]
            # Reshape to (N, C)
            patches = f_norm.squeeze(0).permute(1, 2, 0).reshape(-1, f_norm.shape[1])
            return patches.cpu().numpy(), (feat_h, feat_w)


class AnomalyDetector:
    """
    PatchCore-lite anomaly detector with automatic switching between
    Reference Mode and Self-Reference Mode.
    """

    def __init__(
        self,
        reference_dir: str = "data/reference",
        max_bank_size: int = 50000,
        device: Optional[torch.device] = None,
    ):
        self.reference_dir = reference_dir
        self.max_bank_size = max_bank_size
        self.device = device or get_device()

        os.makedirs(self.reference_dir, exist_ok=True)

        self.feature_extractor = PatchCoreFeatureExtractor(self.device)
        self.memory_bank: Optional[np.ndarray] = None
        self.knn_index: Optional[NearestNeighbors] = None
        self.reference_image_count: int = 0

        # Scan and build reference bank if reference images exist
        self.rebuild_memory_bank()

    def _prepare_tensor(self, img_rgb: np.ndarray) -> torch.Tensor:
        """Convert RGB numpy array to normalized PyTorch tensor."""
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self.device)

    def rebuild_memory_bank(self) -> int:
        """
        Scan reference_dir, extract patch tokens from all reference images,
        and fit the kNN index.
        """
        ref_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            ref_files.extend(glob.glob(os.path.join(self.reference_dir, ext)))

        self.reference_image_count = len(ref_files)

        if not ref_files:
            self.memory_bank = None
            self.knn_index = None
            return 0

        all_patches = []
        for path in ref_files:
            try:
                img = load_image(path)
                t = self._prepare_tensor(img)
                patches, _ = self.feature_extractor.extract_patches(t)
                all_patches.append(patches)
            except Exception:
                continue

        if not all_patches:
            self.memory_bank = None
            self.knn_index = None
            return 0

        combined = np.vstack(all_patches)

        # Subsample to max_bank_size if needed (uniform random coreset approximation)
        if len(combined) > self.max_bank_size:
            indices = np.random.choice(len(combined), size=self.max_bank_size, replace=False)
            self.memory_bank = combined[indices]
        else:
            self.memory_bank = combined

        # Fit kNN index
        self.knn_index = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        self.knn_index.fit(self.memory_bank)
        return len(self.memory_bank)

    def clear_references(self) -> None:
        """Clear all reference images and reset to self-reference mode."""
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            for f in glob.glob(os.path.join(self.reference_dir, ext)):
                try:
                    os.remove(f)
                except Exception:
                    pass
        self.memory_bank = None
        self.knn_index = None
        self.reference_image_count = 0

    def add_reference_image(self, img_source: Any, filename: Optional[str] = None) -> int:
        """Add a reference image to data/reference/ and rebuild memory bank."""
        img = load_image(img_source)
        if not filename:
            filename = f"ref_{len(glob.glob(os.path.join(self.reference_dir, '*'))):03d}.png"

        save_path = os.path.join(self.reference_dir, filename)
        # Convert RGB to BGR for cv2
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return self.rebuild_memory_bank()

    def get_status(self) -> Dict[str, Any]:
        """Return operational status and metadata for UI chips."""
        has_ref = self.knn_index is not None and self.reference_image_count > 0
        return {
            "mode": f"REFERENCE ({self.reference_image_count} IMG)" if has_ref else "SELF-REFERENCE",
            "is_reference_mode": has_ref,
            "reference_count": self.reference_image_count,
            "bank_size": len(self.memory_bank) if self.memory_bank is not None else 0,
        }

    def detect(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Compute anomaly score map for query image.
        Uses reference kNN if available, otherwise calculates self-reference outlier score.
        """
        orig_h, orig_w = img_rgb.shape[:2]
        tensor = self._prepare_tensor(img_rgb)
        query_patches, (feat_h, feat_w) = self.feature_extractor.extract_patches(tensor)

        status_info = self.get_status()
        mode_used = status_info["mode"]

        if self.knn_index is not None and self.reference_image_count > 0:
            # --- Reference Mode (PatchCore kNN distance) ---
            distances, _ = self.knn_index.kneighbors(query_patches, n_neighbors=1)
            raw_scores = distances.squeeze()
        else:
            # --- Self-Reference Mode (Within-Image Statistical Outlier Scoring) ---
            # 1. Distance to median/centroid of the terrain
            centroid = np.median(query_patches, axis=0, keepdims=True)
            centroid_dist = np.linalg.norm(query_patches - centroid, axis=1)

            # 2. Local patch density / k-th nearest neighbor within same image
            k = min(8, max(3, len(query_patches) // 50))
            self_knn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
            self_knn.fit(query_patches)
            dists, _ = self_knn.kneighbors(query_patches)
            # Distance to the k-th neighbor reflects how isolated/rare the patch is
            knn_dist = dists[:, -1]

            # Combine centroid distance and local sparsity
            raw_scores = 0.5 * centroid_dist + 0.5 * knn_dist

        # Reshape to spatial patch grid
        score_grid = raw_scores.reshape(feat_h, feat_w)

        # Resize to original image dimensions
        resized_map = cv2.resize(score_grid, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        # Smooth with Gaussian filter (sigma=4.0) to remove high-frequency tile artifacts
        smoothed = cv2.GaussianBlur(resized_map, (17, 17), sigmaX=4.0)
        from terrascan.utils import suppress_border_margins, robust_saliency_norm
        smoothed = suppress_border_margins(smoothed, margin_ratio=0.03)
        norm_map = robust_saliency_norm(smoothed, noise_percentile=65.0, peak_percentile=99.2)

        return {
            "score_map": norm_map,
            "mode": mode_used,
            "status": status_info,
            "max_score": float(np.max(norm_map)),
            "mean_score": float(np.mean(norm_map)),
        }
