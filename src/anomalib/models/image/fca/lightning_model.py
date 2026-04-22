# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FCA Lightning Model.

Zero-shot texture anomaly detection via Feature Correspondence Analysis.
No training required — call ``engine.test()`` directly.

Example:
    >>> from anomalib.models import Fca
    >>> model = Fca()

    Config::

        model:
          class_path: anomalib.models.Fca
          init_args:
            tile_size: [9, 9]
            sigma_p: 3.0
            sigma_s: 1.0

Reference:
    Ardelean, A-T., & Weyrich, T. (2024). High-Fidelity Zero-Shot Texture
    Anomaly Localization Using Feature Correspondence Analysis. WACV 2024.
    https://github.com/TArdelean/AnomalyLocalizationFCA
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import FCAModel

__all__ = ["Fca"]


class Fca(AnomalibModule):
    """FCA Lightning Module for zero-shot texture anomaly detection.

    FCA requires no training — it computes anomaly maps purely from the
    internal patch statistics of each test image. Use ``engine.test()``
    directly without ``engine.fit()``.

    Args:
        tile_size (tuple[int, int]): Patch size for FCA comparison.
            Larger patches capture more context but reduce spatial
            resolution. Default ``(9, 9)``.
        sigma_p (float): Gaussian sigma for patch-level weighting.
            Controls the spatial extent of feature comparison.
            Default ``3.0``.
        sigma_s (float | None): Sigma for optional post-comparison
            local smoothing blur. Set to ``None`` to disable.
            Default ``1.0``.
        pre_processor (PreProcessor | bool): Pre-processor instance or
            flag to use default. Defaults to ``True``.
        post_processor (PostProcessor | bool): Post-processor instance or
            flag to use default. Defaults to ``True``.
        evaluator (Evaluator | bool): Evaluator instance or flag.
            Defaults to ``True``.
        visualizer (Visualizer | bool): Visualizer instance or flag.
            Defaults to ``True``.

    Example:
        >>> model = Fca(tile_size=(9, 9), sigma_p=3.0, sigma_s=1.0)
        >>> from anomalib.engine import Engine
        >>> engine = Engine()
        >>> engine.test(model, datamodule)
    """

    def __init__(
        self,
        tile_size: tuple[int, int] = (9, 9),
        sigma_p: float = 3.0,
        sigma_s: float | None = 1.0,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        self.model = FCAModel(
            tile_size=tile_size,
            sigma_p=sigma_p,
            sigma_s=sigma_s,
        )

    @property
    def learning_type(self) -> LearningType:
        """FCA is zero-shot — no training data required."""
        return LearningType.ZERO_SHOT

    @property
    def trainer_arguments(self) -> dict:
        """No training needed."""
        return {}

    @staticmethod
    def configure_optimizers() -> None:
        """No optimizer needed for zero-shot inference."""
        return None

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure default pre-processor.

        Uses standard ImageNet normalization. FCA works at the feature level
        so image resolution is flexible — defaults to 256×256 if not specified.

        Args:
            image_size: Target image size. Defaults to ``(256, 256)``.

        Returns:
            PreProcessor: Configured pre-processor.
        """
        #h, w = image_size if image_size is not None else (256, 256)
        #transform = Compose([
        #    Resize((h, w), antialias=True),
        #    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #])
        return PreProcessor(transform=None)

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """Configure default post-processor.

        Returns:
            PostProcessor: Default post-processor instance.
        """
        return PostProcessor()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Compute anomaly maps for a batch of images.

        Args:
            image: Input tensor ``(B, C, H, W)``.

        Returns:
            Anomaly maps ``(B, H, W)``.
        """
        return self.model(image)

    def validation_step(self, batch: Batch, *args, **kwargs) -> dict:
        """Validation step — same as test step.

        Args:
            batch: Anomalib image batch.

        Returns:
            Updated batch with anomaly_map and pred_score.
        """
        del args, kwargs
        anomaly_map = self.model(batch.image)
        pred_score = anomaly_map.flatten(start_dim=1).max(dim=1).values
        return batch.update(anomaly_map=anomaly_map, pred_score=pred_score)

    def test_step(self, batch: Batch, *args, **kwargs) -> dict:
        """Test step — run FCA inference on a batch.

        Args:
            batch: Anomalib image batch.

        Returns:
            Updated batch with anomaly_map and pred_score.
        """
        del args, kwargs
        anomaly_map = self.model(batch.image)
        pred_score = anomaly_map.flatten(start_dim=1).max(dim=1).values
        return batch.update(anomaly_map=anomaly_map, pred_score=pred_score)