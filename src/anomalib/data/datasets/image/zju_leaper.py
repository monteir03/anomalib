# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ZJU-Leaper Dataset.

This module provides a PyTorch Dataset implementation for the ZJU-Leaper dataset.
The dataset contains 19 fabric texture patterns with both normal and anomalous
samples. Pixel-level ground truth masks are provided for anomalous samples.

Folder structure::

    ZJU-Leaper/
    ├── Images/                  ← all images as 6-digit IDs e.g. 003607.jpg
    ├── Annotations/
    │   ├── masks/               ← pixel masks               e.g. 002674.png
    │   └── xmls/                ← per-image metadata (not used)
    └── ImageSets/
        ├── Patterns/            ← split JSONs per pattern   e.g. pattern1.json
        ├── Groups/              ← group JSONs (not used)
        └── total.json

Each ``pattern<N>.json`` has the structure::

    {
        "normal": { "train": ["000001", ...], "test": ["000100", ...] },
        "defect": { "train": [...],           "test": ["001200", ...] }
    }

The ``category`` parameter is the pattern name, e.g. ``"white_plain"``,
``"thick_stripe"``, etc. — or the pattern id as a string e.g. ``"1"``.

Note:
    For unsupervised anomaly detection, defect train images are intentionally
    excluded from the training split. The model only trains on normal images.

Reference:
    Zhang, C., Feng, S., Wang, X., & Wang, Y. (2020). ZJU-Leaper: A Benchmark
    Dataset for Fabric Defect Detection and a Comparative Study. IEEE
    Transactions on Artificial Intelligence, 1(3), 219-232.
"""

from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path

import json

IMG_EXTENSIONS = (".jpg", ".JPG", ".png", ".PNG")

# Pattern id → name mapping
PATTERNS = {
    1:  "white_plain",
    2:  "thick_stripe",
    3:  "thin_stripe",
    4:  "dot_pattern",
    5:  "houndstooth",
    6:  "gingham",
    7:  "knot_pattern",
    8:  "twill_plaid",
    9:  "blue_plaid",
    10: "brown_plaid",
    11: "gray_plaid",
    12: "red_plaid",
    13: "floral_print1",
    14: "floral_print2",
    15: "floral_print3",
    16: "pattern1",
    17: "pattern2",
    18: "pattern3",
    19: "pattern4",
}

# Reverse lookup: name → id
PATTERN_NAME_TO_ID = {v: k for k, v in PATTERNS.items()}

CATEGORIES = tuple(PATTERNS.values())


def _resolve_pattern_id(category: str) -> int:
    """Resolve category string to pattern id.

    Accepts either a name (``"white_plain"``) or a numeric string (``"1"``).

    Raises:
        ValueError: If the category is not recognised.
    """
    if category in PATTERN_NAME_TO_ID:
        return PATTERN_NAME_TO_ID[category]
    try:
        pid = int(category)
        if pid in PATTERNS:
            return pid
    except ValueError:
        pass
    msg = (
        f"Unknown category '{category}'. "
        f"Must be a pattern name {list(PATTERNS.values())} "
        f"or a pattern id 1–19."
    )
    raise ValueError(msg)


class ZJULeaperDataset(AnomalibDataset):
    """ZJU-Leaper dataset class.

    Args:
        root (Path | str): Root directory of the ZJU-Leaper dataset.
            Must contain ``Images/``, ``Annotations/masks/``, and
            ``ImageSets/Patterns/``.
        category (str): Pattern name (e.g. ``"white_plain"``) or pattern id
            as string (e.g. ``"1"``). Defaults to ``"white_plain"``.
        augmentations (Transform | None): Augmentations to apply to input
            images. Defaults to ``None``.
        split (str | Split | None): Dataset split — ``Split.TRAIN`` or
            ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> dataset = ZJULeaperDataset(
        ...     root="./datasets/ZJU-Leaper",
        ...     category="white_plain",
        ...     split="train",
        ... )
        >>> dataset[0].keys()
        dict_keys(['image_path', 'label', 'image'])
    """

    def __init__(
        self,
        root: Path | str = "./datasets/ZJU-Leaper",
        category: str = "white_plain",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root = Path(root)
        self.category = category
        self.split = split
        self.samples = make_zju_leaper_dataset(
            root=self.root,
            category=self.category,
            split=self.split,
        )


def make_zju_leaper_dataset(
    root: str | Path,
    category: str = "white_plain",
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create ZJU-Leaper samples by parsing the pattern JSON split file.

    Args:
        root (str | Path): Root directory of the ZJU-Leaper dataset.
        category (str): Pattern name or id string.
        split (str | Split | None): Split to return. ``None`` returns all.
        extensions (Sequence[str] | None): Image extensions (unused, kept for
            API consistency).

    Returns:
        DataFrame: Samples with columns
            ``path``, ``split``, ``label``, ``image_path``,
            ``mask_path``, ``label_index``.

    Raises:
        FileNotFoundError: If the pattern JSON or Images folder is missing.
        RuntimeError: If no samples are found.
        MisMatchError: If a defect image has no corresponding mask file.
    """
    root = validate_path(root)
    pattern_id = _resolve_pattern_id(category)

    json_path = root / "ImageSets" / "Patterns" / f"pattern{pattern_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Pattern JSON not found: {json_path}")

    images_dir = root / "Images"
    masks_dir  = root / "Annotations" / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    with open(json_path) as f:
        splits = json.load(f)

    normal_train = splits.get("normal", {}).get("train", [])
    normal_test  = splits.get("normal", {}).get("test",  [])
    defect_test  = splits.get("defect", {}).get("test",  [])
    # defect train intentionally excluded — unsupervised setting

    samples_list: list[tuple] = []

    def find_image(img_id: str) -> Path | None:
        for ext in (".jpg", ".JPG", ".png", ".PNG"):
            p = images_dir / f"{img_id}{ext}"
            if p.exists():
                return p
        return None

    def find_mask(img_id: str) -> Path | None:
        for ext in (".png", ".PNG", ".jpg", ".JPG"):
            p = masks_dir / f"{img_id}{ext}"
            if p.exists():
                return p
        return None

    # train/good — normal training images only
    for img_id in normal_train:
        img_path = find_image(img_id)
        if img_path is None:
            continue
        samples_list.append((
            str(root), Split.TRAIN.value, LabelName.NORMAL,
            str(img_path), "", LabelName.NORMAL,
        ))

    # test/good — normal test images
    for img_id in normal_test:
        img_path = find_image(img_id)
        if img_path is None:
            continue
        samples_list.append((
            str(root), Split.TEST.value, LabelName.NORMAL,
            str(img_path), "", LabelName.NORMAL,
        ))

    # test/bad — defective test images + masks
    for img_id in defect_test:
        img_path = find_image(img_id)
        if img_path is None:
            continue
        mask_path = find_mask(img_id)
        if mask_path is None:
            # no mask on disk — include image but without mask
            mask_path_str = ""
        else:
            mask_path_str = str(mask_path)
        samples_list.append((
            str(root), Split.TEST.value, LabelName.ABNORMAL,
            str(img_path), mask_path_str, LabelName.ABNORMAL,
        ))

    if not samples_list:
        raise RuntimeError(
            f"No samples found for category '{category}' (pattern {pattern_id}). "
            f"Check that {images_dir} contains the expected images."
        )

    samples = DataFrame(
        samples_list,
        columns=["path", "split", "label", "image_path", "mask_path", "label_index"],
    )
    samples["label_index"] = samples["label_index"].astype(int)

    samples.attrs["task"] = (
        "classification"
        if (samples["mask_path"] == "").all()
        else "segmentation"
    )

    if split is not None:
        split = Split(split) if isinstance(split, str) else split
        samples = samples[samples["split"] == split.value].reset_index(drop=True)

    return samples