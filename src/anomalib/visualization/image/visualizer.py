# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Image visualization module for anomaly detection.

This module provides the ``ImageVisualizer`` class for visualizing images and their
associated anomaly detection results. The key components include:

    - Visualization of individual fields (images, masks, anomaly maps)
    - Overlay of multiple fields
    - Configurable visualization parameters
    - Support for saving visualizations

Example:
    >>> from anomalib.visualization.image import ImageVisualizer
    >>> # Create visualizer with default settings
    >>> visualizer = ImageVisualizer()
    >>> # Generate visualization
    >>> vis_result = visualizer.visualize(predictions)

The module ensures consistent visualization by:
    - Providing standardized field configurations
    - Supporting flexible overlay options
    - Handling text annotations
    - Maintaining consistent output formats

Note:
    All visualization functions preserve the input image format and dimensions
    unless explicitly specified in the configuration.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image
import csv #new
#from pprint import pprint

# Only import types during type checking to avoid circular imports
if TYPE_CHECKING:
    from lightning.pytorch import Trainer

    from anomalib.data import ImageBatch, ImageItem, NumpyImageBatch, NumpyImageItem
    from anomalib.models import AnomalibModule

from anomalib.utils.path import generate_output_filename
from anomalib.visualization.base import Visualizer
from anomalib.metrics import AUROC, AUPR, F1Max, F1Score, PRO #NEW

from .item_visualizer import (
    DEFAULT_FIELDS_CONFIG,
    DEFAULT_OVERLAY_FIELDS_CONFIG,
    DEFAULT_TEXT_CONFIG,
    visualize_image_item,
)


class ImageVisualizer(Visualizer):
    """Image Visualizer.

    This class visualizes images and their corresponding anomaly maps during testing and
    prediction phases of an anomaly detection model.

    Args:
        fields (list[str] | None, optional): List of fields to visualize.
            Defaults to ``["image", "gt_mask"]``.
        overlay_fields (list[tuple[str, list[str]]] | None, optional): List of tuples
            specifying fields to overlay. Each tuple contains a base field and list of
            fields to overlay on it.
            Defaults to ``[("image", ["anomaly_map"]), ("image", ["pred_mask"])]``.
        field_size (tuple[int, int], optional): Size of each field in visualization as
            ``(width, height)``. Defaults to ``(256, 256)``.
        fields_config (dict[str, dict[str, Any]] | None, optional): Custom configurations
            for field visualization. Merged with ``DEFAULT_FIELDS_CONFIG``.
            Defaults to ``None``.
        overlay_fields_config (dict[str, dict[str, Any]] | None, optional): Custom
            configurations for field overlays. Merged with
            ``DEFAULT_OVERLAY_FIELDS_CONFIG``. Defaults to ``None``.
        text_config (dict[str, Any] | None, optional): Configuration for text overlay.
            Merged with ``DEFAULT_TEXT_CONFIG``. Defaults to ``None``.
        output_dir (str | Path | None, optional): Directory to save visualizations.
            Defaults to ``None``.

    Examples:
        Basic usage with default settings:

        >>> visualizer = ImageVisualizer()

        Customize fields to visualize:

        >>> visualizer = ImageVisualizer(
        ...     fields=["image", "gt_mask", "anomaly_map"],
        ...     overlay_fields=[("image", ["anomaly_map"])]
        ... )

        Adjust field size:

        >>> visualizer = ImageVisualizer(field_size=(512, 512))

        Customize anomaly map visualization:

        >>> fields_config = {
        ...     "anomaly_map": {"colormap": True, "normalize": True}
        ... }
        >>> visualizer = ImageVisualizer(fields_config=fields_config)

        Modify overlay appearance:

        >>> overlay_config = {
        ...     "pred_mask": {"alpha": 0.7, "color": (255, 0, 0), "mode": "fill"},
        ...     "anomaly_map": {"alpha": 0.5, "color": (0, 255, 0), "mode": "contour"}
        ... }
        >>> visualizer = ImageVisualizer(overlay_fields_config=overlay_config)

        Customize text overlay:

        >>> text_config = {
        ...     "font": "arial.ttf",
        ...     "size": 20,
        ...     "color": "yellow",
        ...     "background": (0, 0, 0, 200)
        ... }
        >>> visualizer = ImageVisualizer(text_config=text_config)

        Specify output directory:

        >>> visualizer = ImageVisualizer(output_dir="./output/visualizations")

        Advanced configuration combining multiple customizations:

        >>> visualizer = ImageVisualizer(
        ...     fields=["image", "gt_mask", "anomaly_map", "pred_mask"],
        ...     overlay_fields=[
        ...         ("image", ["anomaly_map"]),
        ...         ("image", ["pred_mask"])
        ...     ],
        ...     field_size=(384, 384),
        ...     fields_config={
        ...         "anomaly_map": {"colormap": True, "normalize": True},
        ...         "pred_mask": {"color": (0, 0, 255)}
        ...     },
        ...     overlay_fields_config={
        ...         "anomaly_map": {"alpha": 0.6, "mode": "fill"},
        ...         "pred_mask": {"alpha": 0.7, "mode": "contour"}
        ...     },
        ...     text_config={
        ...         "font": "times.ttf",
        ...         "size": 24,
        ...         "color": "white",
        ...         "background": (0, 0, 0, 180)
        ...     },
        ...     output_dir="./custom_visualizations"
        ... )

    Note:
        - The ``fields`` parameter determines which individual fields are visualized
        - The ``overlay_fields`` parameter specifies which fields should be overlaid
          on others
        - Field configurations in ``fields_config`` affect how individual fields are
          visualized
        - Overlay configurations in ``overlay_fields_config`` determine how fields are
          blended when overlaid
        - Text configurations in ``text_config`` control the appearance of text labels
          on visualizations
        - If ``output_dir`` is not specified, visualizations will be saved in a
          default location

    For more details on available options for each configuration, refer to the
    documentation of the :func:`visualize_image_item`, :func:`visualize_field`, and
    related functions.
    """

    def __init__(
        self,
        fields: list[str] | None = None,
        overlay_fields: list[tuple[str, list[str]]] | None = None,
        field_size: tuple[int, int] = (256, 256),
        fields_config: dict[str, dict[str, Any]] | None = None,
        overlay_fields_config: dict[str, dict[str, Any]] | None = None,
        text_config: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
        metrics_csv: str | Path | None = None,  # NEW

    ) -> None:
        super().__init__()
        self.fields = fields or ["image", "gt_mask"]
        self.overlay_fields = overlay_fields or [("image", ["anomaly_map"]), ("image", ["pred_mask"])]
        self.field_size = field_size
        self.fields_config = {**DEFAULT_FIELDS_CONFIG, **(fields_config or {})}
        self.overlay_fields_config = {**DEFAULT_OVERLAY_FIELDS_CONFIG, **(overlay_fields_config or {})}
        self.text_config = {**DEFAULT_TEXT_CONFIG, **(text_config or {})}
        self.output_dir = output_dir

        #  New
        self.metrics_csv = Path(metrics_csv) if metrics_csv else None
        if self.metrics_csv:
            self._init_metrics_csv()

    #new
    def _init_metrics_csv(self):
        self.metrics_csv.parent.mkdir(parents=True, exist_ok=True)  # add this line
        if not self.metrics_csv.exists():
            with open(self.metrics_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "image_path",
                    "vis_image_path",
                    "pred_score",
                    "pred_label",
                    "pixel_auroc",
                    "pixel_aupr",
                    "pixel_f1max",
                    "pixel_f1score",
                    "pixel_pro",
                    "loss",
                ])

    # New function to save per image metrics to CSV. Called from on_test_batch_end after visualization and saving.
    def _compute_and_save_metrics(self, item, vis_filename: Path | None, loss: float | None = None):
        """
        Computes and saves per-image pixel-level metrics to CSV.
        Threshold-free metrics (always reliable):
            pixel_auroc, pixel_aupr, pixel_f1max
        Threshold-dependent metrics (reliable when gap = pixel_f1max - pixel_f1score is small):
            pixel_f1score, pixel_pro
        Normal images (gt_mask all zeros) write empty pixel metrics.
        """
        if not (hasattr(item, "gt_mask") and item.gt_mask is not None
                and hasattr(item, "anomaly_map") and item.anomaly_map is not None):
            return

        device = item.anomaly_map.device

        # normal images — pixel metrics undefined without positive gt pixels
        if item.gt_mask.sum() == 0:
            if self.metrics_csv:
                with open(self.metrics_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        str(item.image_path or "unknown"),
                        str(vis_filename) if vis_filename else "none",
                        float(item.pred_score) if item.pred_score is not None else "",
                        str(item.pred_label) if item.pred_label is not None else "",
                        "", "", "", "", "",  # pixel metrics empty
                        loss if loss is not None else "",
                    ])
            return

        results = {}

        # threshold-free metrics
        for metric_cls, key, fields in [
            (AUROC,  "pixel_auroc", ["anomaly_map", "gt_mask"]),
            (AUPR,   "pixel_aupr",  ["anomaly_map", "gt_mask"]),
            (F1Max,  "pixel_f1max", ["anomaly_map", "gt_mask"]),
        ]:
            try:
                m = metric_cls(fields=fields).to(device)
                m.update(item)
                results[key] = float(m.compute())
                m.reset()
            except Exception as e:
                print(f"Erro ao calcular {key} para {item.image_path or 'unknown'}: {e}")
                results[key] = None

        # threshold-dependent — F1Score
        for metric_cls, key, fields in [
            (F1Score, "pixel_f1score", ["pred_mask", "gt_mask"]),
        ]:
            try:
                m = metric_cls(fields=fields).to(device)
                m.update(item)
                results[key] = float(m.compute())
                m.reset()
            except Exception as e:
                print(f"Erro ao calcular {key} para {item.image_path or 'unknown'}: {e}")
                results[key] = None

        # threshold-dependent — PRO (needs explicit shape handling for single image)
        try:
            pred_mask = item.pred_mask
            gt_mask   = item.gt_mask
            if pred_mask is not None and gt_mask is not None:
                # ensure (1, 1, H, W) format
                if pred_mask.ndim == 2:
                    pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)
                elif pred_mask.ndim == 3:
                    pred_mask = pred_mask.unsqueeze(0)
                if gt_mask.ndim == 2:
                    gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
                elif gt_mask.ndim == 3:
                    gt_mask = gt_mask.unsqueeze(0)
                from anomalib.metrics.pro import _PRO
                m = _PRO().to(device)
                m.update(pred_mask.float(), gt_mask.float())
                results["pixel_pro"] = float(m.compute())
        except Exception as e:
            print(f"Erro ao calcular pixel_pro para {item.image_path or 'unknown'}: {e}")
            results["pixel_pro"] = None

        if self.metrics_csv:
            with open(self.metrics_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    str(item.image_path or "unknown"),
                    str(vis_filename) if vis_filename else "none",
                    float(item.pred_score) if item.pred_score is not None else "",
                    str(item.pred_label) if item.pred_label is not None else "",
                    results.get("pixel_auroc",   ""),
                    results.get("pixel_aupr",    ""),
                    results.get("pixel_f1max",   ""),
                    results.get("pixel_f1score", ""),
                    results.get("pixel_pro",     ""),
                    loss if loss is not None else "",
                ])


    def visualize(
        self,
        predictions: "ImageItem | NumpyImageItem | ImageBatch | NumpyImageBatch",
    ) -> Image.Image | list[Image.Image | None] | None:
        """Visualize image predictions.

        This method visualizes anomaly detection predictions intelligently:
        - For single items or single-item batches: returns a single image
        - For multi-item batches: returns a list of images

        Args:
            predictions: The image prediction(s) to visualize. Can be:
                - ``ImageItem``: Single torch-based image item
                - ``NumpyImageItem``: Single numpy-based image item
                - ``ImageBatch``: Batch of torch-based image items
                - ``NumpyImageBatch``: Batch of numpy-based image items

        Returns:
            - For single items or single-item batches: ``Image.Image`` or ``None``
            - For multi-item batches: ``list[Image.Image | None]``

        Examples:
            Visualize a torch-based image item:

            >>> from anomalib.data import ImageItem
            >>> import torch
            >>> item = ImageItem(
            ...     image=torch.rand(3, 224, 224),
            ...     anomaly_map=torch.rand(224, 224),
            ...     pred_mask=torch.rand(224, 224) > 0.5
            ... )
            >>> visualizer = ImageVisualizer()
            >>> result = visualizer.visualize(item)
            >>> isinstance(result, Image.Image) or result is None
            True

            Visualize a numpy-based image item:

            >>> from anomalib.data import NumpyImageItem
            >>> import numpy as np
            >>> item = NumpyImageItem(
            ...     image=np.random.rand(224, 224, 3),
            ...     anomaly_map=np.random.rand(224, 224),
            ...     pred_mask=np.random.rand(224, 224) > 0.5
            ... )
            >>> result = visualizer.visualize(item)
            >>> isinstance(result, Image.Image) or result is None
            True

            Visualize a batch with one image (returns single image, not list):

            >>> from anomalib.data import ImageBatch
            >>> single_batch = ImageBatch(
            ...     image=torch.rand(1, 3, 224, 224),
            ...     anomaly_map=torch.rand(1, 224, 224)
            ... )
            >>> result = visualizer.visualize(single_batch)
            >>> isinstance(result, Image.Image) or result is None
            True

            Visualize a batch with multiple images (returns list):

            >>> multi_batch = ImageBatch(
            ...     image=torch.rand(3, 3, 224, 224),
            ...     anomaly_map=torch.rand(3, 224, 224)
            ... )
            >>> results = visualizer.visualize(multi_batch)
            >>> isinstance(results, list) and len(results) == 3
            True

        Note:
            - The method uses the same configuration (fields, overlays, etc.) as specified
              during initialization of the ``ImageVisualizer``.
            - If an item cannot be visualized (e.g., missing required fields), the
              corresponding result will be ``None``.
            - This method now behaves identically to the ``__call__`` method.
        """
        # Import here to avoid circular imports
        from anomalib.data import ImageBatch, ImageItem, NumpyImageBatch, NumpyImageItem

        # Handle single items
        if isinstance(predictions, (ImageItem, NumpyImageItem)):
            return visualize_image_item(
                predictions,
                fields=self.fields,
                overlay_fields=self.overlay_fields,
                field_size=self.field_size,
                fields_config=self.fields_config,
                overlay_fields_config=self.overlay_fields_config,
                text_config=self.text_config,
            )

        # Handle batches
        if isinstance(predictions, (ImageBatch, NumpyImageBatch)):
            batch_size = len(predictions)

            # Single-item batch - return single image
            if batch_size == 1:
                image_item = next(iter(predictions))
                return visualize_image_item(
                    image_item,  # type: ignore[arg-type]
                    fields=self.fields,
                    overlay_fields=self.overlay_fields,
                    field_size=self.field_size,
                    fields_config=self.fields_config,
                    overlay_fields_config=self.overlay_fields_config,
                    text_config=self.text_config,
                )

            # Multi-item batch - return list of images
            results = []
            for image_item in predictions:
                visualization = visualize_image_item(
                    image_item,  # type: ignore[arg-type]
                    fields=self.fields,
                    overlay_fields=self.overlay_fields,
                    field_size=self.field_size,
                    fields_config=self.fields_config,
                    overlay_fields_config=self.overlay_fields_config,
                    text_config=self.text_config,
                )
                results.append(visualization)
            return results

        msg = (
            f"Unsupported input type: {type(predictions)}. "
            "Expected ImageItem, NumpyImageItem, ImageBatch, or NumpyImageBatch."
        )
        raise TypeError(msg)

    def __call__(
        self,
        predictions: "ImageItem | NumpyImageItem | ImageBatch | NumpyImageBatch",
    ) -> Image.Image | list[Image.Image | None] | None:
        """Make the visualizer callable.

        This method allows the visualizer to be used as a callable object.
        It behaves identically to the ``visualize`` method.

        Args:
            predictions: The predictions to visualize. Same as ``visualize``.

        Returns:
            Same as ``visualize`` method.

        Examples:
            >>> visualizer = ImageVisualizer()
            >>> result = visualizer(predictions)  # Equivalent to visualizer.visualize(predictions)
        """
        return self.visualize(predictions)

    def on_test_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "AnomalibModule",
        outputs: "ImageBatch",
        batch: "ImageBatch",
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends."""
        del outputs, batch_idx, dataloader_idx #pl_module # Unused arguments.

        if self.output_dir is None:
            self.output_dir = Path(trainer.default_root_dir) / "images"

        # read loss cache from model — None if model doesn't implement test_step
        loss_per_image = getattr(pl_module, "_test_loss_cache", None)

        #first = True

        #here we iterating over batch to compute and save metrics per image.
        for i,item in enumerate(batch):
            #if first:
                #print("#######DEBUG_ITEM########")
                ##pprint(dir(item))
                #first = False

            image = visualize_image_item(
                item,
                fields=self.fields,
                overlay_fields=self.overlay_fields,
                field_size=self.field_size,
                fields_config=self.fields_config,
                overlay_fields_config=self.overlay_fields_config,
                text_config=self.text_config,
            )

            filename = None
            if image is not None:
                # Get the dataset name and category to save the image
                datamodule = getattr(trainer, "datamodule", None)
                dataset_name = getattr(datamodule, "name", None) if datamodule else None
                category = getattr(datamodule, "category", None) if datamodule else None

                # For datasets with a flat image folder (e.g. ZJU-Leaper where all
                # images live in Images/), inject a good/bad subfolder based on
                # gt_label so normal and anomalous visualizations are separated.
                # For MVTec-style datasets the input path already encodes the
                # subfolder (test/good/, test/crack/, test/0/ etc.) so we leave
                # those untouched.
                from anomalib.data.utils import LabelName
                input_path = item.image_path or ""
                output_path = self.output_dir
                parent_name = Path(input_path).parent.name
                is_flat = parent_name not in ("good", "bad", "0") and not any(
                    part in ("test", "train") for part in Path(input_path).parts
                )
                if is_flat and hasattr(item, "gt_label") and item.gt_label is not None:
                    label_folder = "bad" if int(item.gt_label) == LabelName.ABNORMAL else "good"
                    output_path = self.output_dir / label_folder

                filename = generate_output_filename(
                    input_path=input_path,
                    output_path=output_path,
                    dataset_name=dataset_name,
                    category=category,
                )

                # Save the image to the specified filename
                image.save(filename)
            
            #NEW. main modification
            # get loss for this specific image — None if model doesn't support it
            item_loss = loss_per_image[i].item() if loss_per_image is not None else None
            self._compute_and_save_metrics(item, filename, loss=item_loss)

    def on_predict_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "AnomalibModule",
        outputs: "ImageBatch",
        batch: "ImageBatch",
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the predict batch ends."""
        return self.on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
