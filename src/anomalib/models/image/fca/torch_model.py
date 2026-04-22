# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FCA Torch Model.

Feature Correspondence Analysis (FCA) for zero-shot texture anomaly detection.
No training required — anomaly maps are computed purely from internal image
statistics by comparing patch features against a global texture reference.

Reference:
    Ardelean, A-T., & Weyrich, T. (2024). High-Fidelity Zero-Shot Texture
    Anomaly Localization Using Feature Correspondence Analysis. WACV 2024.
    https://github.com/TArdelean/AnomalyLocalizationFCA
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


# ── Feature Extractor ─────────────────────────────────────────────────────────

class WideResnetExtractor(nn.Module):
    """WideResNet-50 feature extractor (first 6 blocks).

    Extracts mid-level features suitable for texture analysis.
    Features are post-scaled to zero mean and unit variance per channel.
    """

    def __init__(self) -> None:
        super().__init__()
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.normalization_std  = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        cnn = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval()
        self.backbone = nn.Sequential(*list(cnn.children())[:6])
        # freeze — FCA is zero-shot, no gradient needed through backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        mean = self.normalization_mean.to(image.device)
        std  = self.normalization_std.to(image.device)
        return (image - mean) / std

    @staticmethod
    def scale_features(features: torch.Tensor) -> torch.Tensor:
        """Per-channel standardisation (zero mean, unit std)."""
        b, c, h, w = features.shape
        flat = features.view(b, c, -1)
        mean = flat.mean(dim=-1, keepdim=True).unsqueeze(-1)
        std  = flat.std(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-8)
        return (features - mean) / std

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(self.normalize(image))
        return self.scale_features(features)


# ── FCA Core ──────────────────────────────────────────────────────────────────

def _get_gaussian_kernel(
    device: torch.device,
    tile_size: tuple[int, int],
    sigma: float,
) -> torch.Tensor:
    """Build a 2D Gaussian kernel of shape ``(tile_size[0], tile_size[1])``."""
    ky = torch.arange(tile_size[0], device=device, dtype=torch.float32) - tile_size[0] // 2
    kx = torch.arange(tile_size[1], device=device, dtype=torch.float32) - tile_size[1] // 2
    gy, gx = torch.meshgrid(ky, kx, indexing="ij")
    kernel = torch.exp(-(gy ** 2 + gx ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _reflect_pad(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    p = patch_size // 2
    return F.pad(x, (p, p, p, p), mode="reflect")


def _reference_median(
    features: torch.Tensor,
    tile_size: tuple[int, int],
) -> torch.Tensor:
    """Compute median reference patch from the feature map."""
    unf = F.unfold(features, tile_size, stride=tile_size)[0].T
    unf = unf.reshape(-1, features.shape[1], tile_size[0] * tile_size[1])
    val, _ = torch.sort(unf, dim=-1)
    return torch.median(val, dim=0).values  # C × tile²


class FCACore(nn.Module):
    """Core Feature Correspondence Analysis algorithm.

    Computes a per-pixel anomaly score by comparing each patch's sorted
    feature distribution against a global median reference using a
    Gaussian-weighted L1 loss, then aggregates per-pixel contributions
    from all overlapping patches.

    Args:
        tile_size (tuple[int, int]): Patch size for comparison. Default (9, 9).
        chunk_size (int): Number of rows processed at a time. Default 8.
        sigma_p (float): Gaussian sigma for patch weighting. Default 3.0.
        sigma_s (float | None): Sigma for optional local smoothing blur.
            Set to ``None`` to disable. Default 1.0.
        k_s (int): Kernel size for local smoothing blur. Default 5.
    """

    def __init__(
        self,
        tile_size: tuple[int, int] = (9, 9),
        chunk_size: int = 8,
        sigma_p: float = 3.0,
        sigma_s: float | None = 1.0,
        k_s: int = 5,
    ) -> None:
        super().__init__()
        assert tile_size[0] == tile_size[1], "Only square patches supported."
        self.tile_size  = tuple(tile_size)
        self.chunk_size = chunk_size
        self.sigma_p    = sigma_p
        self.p_size     = tile_size[0] // 2
        self.local_blur = (
            torchvision.transforms.GaussianBlur(k_s, sigma=sigma_s)
            if sigma_s is not None else None
        )
        self._gaussian_kernel: torch.Tensor | None = None

    def _get_kernel(self, device: torch.device) -> torch.Tensor:
        if self._gaussian_kernel is None or self._gaussian_kernel.device != device:
            self._gaussian_kernel = _get_gaussian_kernel(
                device, self.tile_size, self.sigma_p
            ).reshape(-1)
        return self._gaussian_kernel

    def _generate_all_sets(self, features: torch.Tensor):
        """Yield row-chunks of unfolded patch sets."""
        b, c, h, w = features.shape
        padded = _reflect_pad(features, self.tile_size[0])
        unf = F.unfold(padded, (self.tile_size[0], padded.shape[-1]), stride=(1, 1))
        unf = unf[0].T.reshape(h, c, self.tile_size[0], padded.shape[-1])
        for i in range(0, h, self.chunk_size):
            chunk = unf[i : i + self.chunk_size]
            unf_2 = F.unfold(chunk, self.tile_size, stride=(1, 1))
            unf_2 = unf_2.transpose(1, 2).reshape(
                chunk.shape[0], w, c, self.tile_size[0] * self.tile_size[1]
            )
            yield unf_2

    @torch.no_grad()
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute FCA anomaly map.

        Args:
            features: Feature tensor of shape ``(1, C, H, W)``.

        Returns:
            Anomaly map of shape ``(H, W)``.
        """
        assert features.shape[0] == 1, "FCA processes one image at a time."
        device = features.device
        r_set  = _reference_median(features, self.tile_size)   # C × tile²
        kernel = self._get_kernel(device)

        wp = features.shape[-1] + 2 * self.p_size
        parts = []

        for f_set in self._generate_all_sets(features):
            # f_set: h_chunk × W × C × tile²
            fvalues, ind = torch.sort(f_set, dim=-1)
            vec_arr = r_set[None, None].expand_as(fvalues)
            diff    = F.l1_loss(fvalues, vec_arr, reduction="none")

            # un-sort to restore spatial correspondence
            diff_re = torch.gather(
                diff, dim=-1, index=torch.argsort(ind)
            ).mean(dim=2, keepdim=True)  # h × W × 1 × tile²

            if self.local_blur is not None:
                diff_re = self.local_blur(
                    diff_re.view(-1, 1, *self.tile_size)
                ).reshape(diff_re.shape)

            diff_re = diff_re * kernel   # h × W × 1 × tile²
            diff_re = diff_re.permute(0, 2, 3, 1).reshape(
                f_set.shape[0], -1, features.shape[-1]
            )  # h × tile² × W
            c_fold = F.fold(
                diff_re, (self.tile_size[0], wp), kernel_size=self.tile_size
            )  # h × 1 × tile × WP
            parts.append(c_fold)

        combined = torch.cat(parts, dim=0)  # H × 1 × tile × WP
        folded = F.fold(
            combined.permute(1, 2, 3, 0).reshape(1, -1, features.shape[-2]),
            output_size=(wp, wp),
            kernel_size=(self.tile_size[0], wp),
        )
        # remove reflect-pad borders
        anomaly_map = folded[0, 0, self.p_size:-self.p_size, self.p_size:-self.p_size]
        return anomaly_map  # H × W


# ── Combined model ────────────────────────────────────────────────────────────

class FCAModel(nn.Module):
    """Full FCA model: feature extraction + anomaly map computation.

    Args:
        tile_size (tuple[int, int]): FCA patch size. Default (9, 9).
        sigma_p (float): Gaussian sigma for patch weighting. Default 3.0.
        sigma_s (float | None): Local smoothing sigma. Default 1.0.
    """

    def __init__(
        self,
        tile_size: tuple[int, int] = (9, 9),
        sigma_p: float = 3.0,
        sigma_s: float | None = 1.0,
    ) -> None:
        super().__init__()
        self.feature_extractor = WideResnetExtractor()
        self.fca = FCACore(
            tile_size=tile_size,
            sigma_p=sigma_p,
            sigma_s=sigma_s,
        )

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Compute anomaly map for a batch of images.

        Processes each image independently (FCA is single-image by design)
        and stacks results into a batch.

        Args:
            image: Input tensor of shape ``(B, C, H, W)`` in [0, 1].

        Returns:
            Anomaly maps of shape ``(B, H, W)``, resized to input resolution.
        """
        b, c, h, w = image.shape
        maps = []
        for i in range(b):
            features = self.feature_extractor(image[i : i + 1])  # 1 × C × H' × W'
            amap = self.fca(features)                              # H' × W'
            # upsample to original image resolution
            amap = F.interpolate(
                amap[None, None], size=(h, w), mode="bilinear", align_corners=False
            )[0, 0]
            maps.append(amap)
        return torch.stack(maps, dim=0)  # B × H × W