# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ZJU-Leaper Lightning DataModule.

This module provides the Lightning DataModule for the ZJU-Leaper dataset,
following the same interface as anomalib's other image datamodules.

Example:
    Using the Python API::

        >>> from anomalib.data import ZJULeaper
        >>> datamodule = ZJULeaper(
        ...     root="./datasets/ZJU-Leaper",
        ...     category="white_plain",
        ... )
        >>> datamodule.setup()

    Using the CLI config::

        class_path: anomalib.data.ZJULeaper
        init_args:
            root: /mnt/data02/data/cache/ZJU-Leaper
            num_workers: 10
            seed: 42
            max_train_samples: 300
            max_test_normal_samples: null  # defaults to match defect count (1:1)
            category:
              grid:
                - white_plain
                - thick_stripe
                - dot_pattern

Note on ``max_train_samples``:
    ZJU-Leaper was designed for supervised defect detection and contains
    up to 3000+ normal training images per category. For unsupervised
    anomaly detection, 300-500 images are sufficient to learn the normal
    distribution — consistent with MVTec-AD which uses ~200-300 images.
    Setting ``max_train_samples=300`` (default) aligns ZJU-Leaper with
    the standard unsupervised protocol and avoids memory issues with
    memory-bank models like Patchcore. Set to ``None`` to use all images.

Note on ``max_test_normal_samples``:
    ZJU-Leaper has up to 1500 normal test images per category vs ~300
    defective test images, creating a heavily imbalanced test set that
    slows inference and inflates AUROC estimates. By default this is set
    to match the number of defective test images (1:1 ratio), which is
    a common choice in anomaly detection benchmarks, gives reliable AUROC
    estimates and keeps inference fast. Set to ``None`` to use all normal
    test images.
"""

import logging
from pathlib import Path

import numpy as np
from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.zju_leaper import ZJULeaperDataset, CATEGORIES
from anomalib.data.utils import LabelName, Split, TestSplitMode, ValSplitMode

logger = logging.getLogger(__name__)


class ZJULeaper(AnomalibDataModule):
    """ZJU-Leaper Lightning DataModule.

    Args:
        root (Path | str): Root directory of the ZJU-Leaper dataset.
            Must contain ``Images/``, ``Annotations/masks/``, and
            ``ImageSets/Patterns/``.
            Defaults to ``"./datasets/ZJU-Leaper"``.
        category (str): Pattern name (e.g. ``"white_plain"``) or pattern id
            as string (e.g. ``"1"``). Defaults to ``"white_plain"``.
        max_train_samples (int | None): Maximum number of normal training
            images to use. ZJU-Leaper contains up to 3000+ normal training
            images per category, which is excessive for unsupervised anomaly
            detection. Setting this to 300-500 aligns with the MVTec-AD
            protocol and avoids memory issues with memory-bank models.
            Set to ``None`` to use all available training images.
            Defaults to ``300``.
        max_test_normal_samples (int | None): Maximum number of normal images
            in the test set. By default (``None``) this is set to match the
            number of defective test images (1:1 ratio), which gives reliable
            AUROC estimates while keeping inference fast. Set to an explicit
            integer to override, or ``-1`` to use all normal test images.
            Defaults to ``None`` (match defect count).
        train_batch_size (int): Training batch size. Defaults to ``32``.
        eval_batch_size (int): Evaluation batch size. Defaults to ``32``.
        num_workers (int): Number of dataloader workers. Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations for training.
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations for validation.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations for testing.
            Defaults to ``None``.
        augmentations (Transform | None): Shared augmentations used when
            stage-specific ones are not provided. Defaults to ``None``.
        test_split_mode (TestSplitMode): How to obtain the test split.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction to use for test when splitting.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): How to obtain the validation split.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction to use for validation when splitting.
            Defaults to ``0.5``.
        seed (int | None): Random seed for reproducibility. Defaults to ``None``.

    Example:
        >>> datamodule = ZJULeaper(
        ...     root="./datasets/ZJU-Leaper",
        ...     category="white_plain",
        ...     max_train_samples=300,
        ...     max_test_normal_samples=None,  # defaults to 1:1 ratio
        ... )
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data["image"].shape
        torch.Size([32, 3, 256, 256])
        >>> i, data = next(enumerate(datamodule.test_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])
    """

    def __init__(
        self,
        root: Path | str = "./datasets/ZJU-Leaper",
        category: str = "white_plain",
        max_train_samples: int | None = 300,
        max_test_normal_samples: int | None = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = category
        self.max_train_samples = max_train_samples
        self.max_test_normal_samples = max_test_normal_samples

    def _subsample(
        self,
        samples,
        n: int,
        label: str = "train",
    ):
        """Randomly subsample rows from a DataFrame using the datamodule seed."""
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(len(samples), size=n, replace=False)
        return samples.iloc[sorted(indices)].reset_index(drop=True)

    def _setup(self, _stage: str | None = None) -> None:
        """Set up train and test datasets.

        Applies ``max_train_samples`` to the training set and
        ``max_test_normal_samples`` (defaulting to 1:1 ratio with defects)
        to the normal test images. Both use the datamodule seed for
        reproducibility.
        """
        # ── training set ──────────────────────────────────────────────────────
        self.train_data = ZJULeaperDataset(
            root=self.root,
            category=self.category,
            augmentations=self.train_augmentations,
            split=Split.TRAIN,
        )

        if (
            self.max_train_samples is not None
            and len(self.train_data.samples) > self.max_train_samples
        ):
            original_n = len(self.train_data.samples)
            self.train_data.samples = self._subsample(
                self.train_data.samples, self.max_train_samples
            )
            logger.info(
                f"ZJULeaper [{self.category}]: train set subsampled "
                f"{original_n} → {self.max_train_samples} normal images."
            )

        # ── test set ──────────────────────────────────────────────────────────
        self.test_data = ZJULeaperDataset(
            root=self.root,
            category=self.category,
            augmentations=self.test_augmentations,
            split=Split.TEST,
        )

        samples = self.test_data.samples
        normal_mask  = samples["label_index"] == LabelName.NORMAL
        defect_mask  = samples["label_index"] == LabelName.ABNORMAL

        normal_samples = samples[normal_mask]
        defect_samples = samples[defect_mask]

        n_defects = len(defect_samples)

        # resolve how many normal test images to keep
        if self.max_test_normal_samples == -1:
            # -1 means keep all
            keep_n = len(normal_samples)
        elif self.max_test_normal_samples is None:
            # default: match defect count (1:1 ratio)
            keep_n = n_defects
        else:
            keep_n = self.max_test_normal_samples

        if len(normal_samples) > keep_n:
            original_n = len(normal_samples)
            normal_samples = self._subsample(normal_samples, keep_n)
            logger.info(
                f"ZJULeaper [{self.category}]: test normal subsampled "
                f"{original_n} → {keep_n} images (1:1 ratio with {n_defects} defects)."
            )

        # reconstruct test samples: subsampled normals + all defects
        import pandas as pd
        self.test_data.samples = (
            pd.concat([normal_samples, defect_samples])
            .sort_values("image_path")
            .reset_index(drop=True)
        )