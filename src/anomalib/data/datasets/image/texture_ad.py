# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Texture-AD Dataset.

This module provides a PyTorch Dataset implementation for the Texture-AD dataset.
The dataset contains three material types (cloth, metal, wafer), each with 15
numbered texture subcategories. Both normal and anomalous samples are provided,
with pixel-level ground truth masks for all anomalous test images.

Folder structure (per material)::

    <root>/
    └── <material>_mvtec_format/      e.g. cloth_mvtec_format/
        ├── meta.json                  (not used — filesystem is parsed directly)
        ├── 1/
        │   ├── train/
        │   │   └── good/             ← normal training images
        │   ├── test/
        │   │   ├── good/             ← normal test images
        │   │   └── 0/                ← defective test images
        │   └── ground_truth/
        │       └── 0/                ← masks (same filename as test/0/ image)
        ├── 2/
        │   └── ...
        └── 15/
            └── ...

The category parameter encodes both material and subcategory id as
``"<material>_<id>"``, e.g. ``"cloth_1"``, ``"metal_3"``, ``"wafer_7"``.
This mirrors how anomalib's other datamodules expose a flat ``category`` string.

Example:
    >>> from anomalib.data.datasets.image.texture_ad import TextureADDataset
    >>> dataset = TextureADDataset(
    ...     root="./datasets/Texture-AD",
    ...     category="cloth_1",
    ...     split="train",
    ... )

Reference:
    Texture-AD: A Texture Anomaly Detection Benchmark for Real-world
    Industrial Applications.
"""

from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path

IMG_EXTENSIONS = (".png", ".PNG")

MATERIALS = ("cloth", "metal", "wafer")

MATERIAL_CATEGORIES = {
    "cloth": list(range(1, 16)),  # 1-15
    "metal": list(range(1, 11)),  # 1-10
    "wafer": list(range(1, 15)),  # 1-14
}

# flat category list: cloth_1…cloth_15, metal_1…metal_10, wafer_1…wafer_14
CATEGORIES = tuple(
    f"{material}_{i}"
    for material, ids in MATERIAL_CATEGORIES.items()
    for i in ids
)

def _parse_category(category: str) -> tuple[str, str]:
    """Split ``"cloth_3"`` → ``("cloth", "3")``.

    Raises:
        ValueError: If the category string is not in ``"<material>_<id>"`` format
            or the material is not one of the known materials.
    """
    parts = category.rsplit("_", 1)
    if len(parts) != 2:
        msg = (
            f"Invalid category '{category}'. "
            f"Expected format: '<material>_<id>', e.g. 'cloth_1'."
        )
        raise ValueError(msg)
    material, cat_id = parts
    if material not in MATERIALS:
        msg = (
            f"Unknown material '{material}' in category '{category}'. "
            f"Available materials: {MATERIALS}."
        )
        raise ValueError(msg)
    return material, cat_id


class TextureADDataset(AnomalibDataset):
    """Texture-AD dataset class.

    Args:
        root (Path | str): Root directory of the Texture-AD dataset.
            Must contain ``<material>_mvtec_format/`` subdirectories.
        category (str): Category in ``"<material>_<id>"`` format,
            e.g. ``"cloth_1"``, ``"metal_3"``, ``"wafer_7"``.
            Defaults to ``"cloth_1"``.
        augmentations (Transform | None): Augmentations to apply to input
            images. Defaults to ``None``.
        split (str | Split | None): Dataset split — ``Split.TRAIN`` or
            ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> dataset = TextureADDataset(
        ...     root="./datasets/Texture-AD",
        ...     category="cloth_3",
        ...     split="train",
        ... )
        >>> dataset[0].keys()
        dict_keys(['image_path', 'label', 'image'])
    """

    def __init__(
        self,
        root: Path | str = "./datasets/Texture-AD",
        category: str = "cloth_1",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        material, cat_id = _parse_category(category)
        # point root_category at e.g. <root>/cloth_mvtec_format/1/
        self.root_category = Path(root) / f"{material}_mvtec_format" / cat_id
        self.category = category
        self.split = split
        self.samples = make_texture_ad_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS,
        )


def make_texture_ad_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create Texture-AD samples by parsing the directory structure.

    The directory layout mirrors MVTec AD exactly::

        <root>/
        ├── train/good/*.png
        ├── test/good/*.png
        ├── test/0/*.png
        └── ground_truth/0/*.png

    The implementation therefore follows ``make_mvtec_ad_dataset`` closely,
    using filesystem globbing and sorted mask matching.

    Args:
        root (str | Path): Path to the category root
            (e.g. ``cloth_mvtec_format/1/``).
        split (str | Split | None): Split to return. ``None`` returns all.
        extensions (Sequence[str] | None): Image file extensions to include.

    Returns:
        DataFrame: Samples with columns
            ``path``, ``split``, ``label``, ``image_path``,
            ``mask_path``, ``label_index``.

    Raises:
        RuntimeError: If no images are found under ``root``.
        MisMatchError: If defective images and masks do not match by filename
            stem.
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)

    # glob all images — yields paths like:
    #   train/good/Image_....png
    #   test/good/Image_....png
    #   test/0/Image_....png
    #   ground_truth/0/Image_....png
    samples_list = [
        (str(root), *f.parts[-3:])
        for f in root.glob("**/*")
        if f.suffix in extensions
    ]
    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = DataFrame(
        samples_list,
        columns=["path", "split", "label", "image_path"],
    )

    # reconstruct absolute image paths
    samples["image_path"] = (
        samples["path"]
        + "/"
        + samples["split"]
        + "/"
        + samples["label"]
        + "/"
        + samples["image_path"]
    )

    # label index: "good" → NORMAL, anything else → ABNORMAL
    samples.loc[samples["label"] == "good", "label_index"] = LabelName.NORMAL
    samples.loc[samples["label"] != "good", "label_index"] = LabelName.ABNORMAL
    samples["label_index"] = samples["label_index"].astype(int)

    # separate ground_truth masks from image samples
    mask_samples = samples.loc[samples["split"] == "ground_truth"].sort_values(
        by="image_path", ignore_index=True
    )
    samples = samples[samples["split"] != "ground_truth"].sort_values(
        by="image_path", ignore_index=True
    )

    # assign mask paths to abnormal test images only
    samples["mask_path"] = ""
    samples.loc[
        (samples["split"] == "test") & (samples["label_index"] == LabelName.ABNORMAL),
        "mask_path",
    ] = mask_samples["image_path"].to_numpy()

    # verify that each mask filename stem contains the image filename stem
    abnormal_samples = samples.loc[samples["label_index"] == LabelName.ABNORMAL]
    if len(abnormal_samples) and not abnormal_samples.apply(
        lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1
    ).all():
        msg = (
            "Mismatch between anomalous images and ground truth masks. "
            "Ensure that mask files in 'ground_truth/' share the same "
            "filename as the corresponding test images."
        )
        raise MisMatchError(msg)

    samples.attrs["task"] = (
        "classification" if (samples["mask_path"] == "").all() else "segmentation"
    )

    if split:
        split = Split(split) if isinstance(split, str) else split
        samples = samples[samples["split"] == split.value].reset_index(drop=True)

    return samples