# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Texture-AD Lightning DataModule.

This module provides the Lightning DataModule for the Texture-AD dataset,
following the same interface as anomalib's other image datamodules (e.g. MVTecAD).

The ``category`` parameter uses the format ``"<material>_<id>"``:

    - ``"cloth_1"``  →  ``cloth_mvtec_format/1/``
    - ``"metal_3"``  →  ``metal_mvtec_format/3/``
    - ``"wafer_7"``  →  ``wafer_mvtec_format/7/``

Example:
    Using the Python API::

        >>> from anomalib.data import TextureAD
        >>> datamodule = TextureAD(
        ...     root="./datasets/Texture-AD",
        ...     category="cloth_1",
        ... )
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data["image"].shape
        torch.Size([32, 3, 256, 256])

    Using the CLI config::

        class_path: anomalib.data.TextureAD
        init_args:
            root: /mnt/data02/data/cache/texture-ad
            category: cloth_1
            num_workers: 10
            seed: 42

    Running all cloth categories via benchmark grid::

        data:
          class_path: anomalib.data.TextureAD
          init_args:
            root: /mnt/data02/data/cache/texture-ad
            num_workers: 10
            seed: 42
          category:
            grid:
              - cloth_1
              - cloth_2
              - cloth_3
              ...
              - metal_1
              - wafer_1
"""

import logging
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.texture_ad import TextureADDataset, CATEGORIES
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode

logger = logging.getLogger(__name__)


class TextureAD(AnomalibDataModule):
    """Texture-AD Lightning DataModule.

    Args:
        root (Path | str): Root directory of the Texture-AD dataset.
            Must contain ``<material>_mvtec_format/`` subdirectories.
            Defaults to ``"./datasets/Texture-AD"``.
        category (str): Category in ``"<material>_<id>"`` format,
            e.g. ``"cloth_1"``, ``"metal_3"``, ``"wafer_7"``.
            Defaults to ``"cloth_1"``.
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
        >>> datamodule = TextureAD(
        ...     root="./datasets/Texture-AD",
        ...     category="cloth_3",
        ... )
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.test_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])
        >>> data["image"].shape, data["mask"].shape
        (torch.Size([32, 3, 256, 256]), torch.Size([32, 256, 256]))
    """

    def __init__(
        self,
        root: Path | str = "./datasets/Texture-AD",
        category: str = "cloth_1",
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

    def _setup(self, _stage: str | None = None) -> None:
        """Set up train and test datasets."""
        self.train_data = TextureADDataset(
            root=self.root,
            category=self.category,
            augmentations=self.train_augmentations,
            split=Split.TRAIN,
        )
        self.test_data = TextureADDataset(
            root=self.root,
            category=self.category,
            augmentations=self.test_augmentations,
            split=Split.TEST,
        )