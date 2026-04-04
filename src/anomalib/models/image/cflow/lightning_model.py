# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CFLOW - Real-Time Unsupervised Anomaly Detection via Conditional Normalizing Flows.

This module implements the CFLOW model for anomaly detection. CFLOW uses conditional
normalizing flows to model the distribution of normal data and detect anomalies in
real-time.

The model consists of:
    - A CNN backbone encoder to extract features
    - Multiple decoders using normalizing flows to model feature distributions
    - Positional encoding to capture spatial information

Paper: `Real-Time Unsupervised Anomaly Detection via Conditional Normalizing Flows
<https://arxiv.org/pdf/2107.12571v1.pdf>`_
"""

__all__ = ["Cflow"]

from collections.abc import Sequence
from typing import Any

import einops
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
from torch.nn import functional as F  # noqa: N812
from torch.optim import Optimizer

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import CflowModel
from .utils import get_logp, positional_encoding_2d


class Cflow(AnomalibModule):
    """PyTorch Lightning implementation of the CFLOW model.

    The model uses a pre-trained CNN backbone to extract features, followed by
    conditional normalizing flow decoders to model the distribution of normal data.

    Args:
        backbone (str, optional): Name of the backbone CNN network.
            Defaults to ``"wide_resnet50_2"``.
        layers (Sequence[str], optional): List of layer names to extract features
            from. Defaults to ``("layer2", "layer3", "layer4")``.
        pre_trained (bool, optional): If True, use pre-trained weights for the
            backbone. Defaults to ``True``.
        fiber_batch_size (int, optional): Batch size for processing individual
            fibers. Defaults to ``64``.
        decoder (str, optional): Type of normalizing flow decoder to use.
            Defaults to ``"freia-cflow"``.
        condition_vector (int, optional): Dimension of the condition vector.
            Defaults to ``128``.
        coupling_blocks (int, optional): Number of coupling blocks in the flow.
            Defaults to ``8``.
        clamp_alpha (float, optional): Clamping value for the alpha parameter in
            flows. Defaults to ``1.9``.
        permute_soft (bool, optional): If True, use soft permutation in flows.
            Defaults to ``False``.
        lr (float, optional): Learning rate for the optimizer.
            Defaults to ``0.0001``.
        pre_processor (PreProcessor | bool, optional): Pre-processing module.
            Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processing module.
            Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluation module.
            Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualization module.
            Defaults to ``True``.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3", "layer4"),
        pre_trained: bool = True,
        fiber_batch_size: int = 64,
        decoder: str = "freia-cflow",
        condition_vector: int = 128,
        coupling_blocks: int = 8,
        clamp_alpha: float = 1.9,
        permute_soft: bool = False,
        lr: float = 0.0001,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model: CflowModel = CflowModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            fiber_batch_size=fiber_batch_size,
            decoder=decoder,
            condition_vector=condition_vector,
            coupling_blocks=coupling_blocks,
            clamp_alpha=clamp_alpha,
            permute_soft=permute_soft,
        )
        self.automatic_optimization = False
        # TODO(ashwinvaidya17): LR should be part of optimizer in config.yaml since  # noqa: TD003
        # cflow has custom optimizer. CVS-122670
        self.learning_rate = lr

    def configure_optimizers(self) -> Optimizer:
        """Configure optimizers for each decoder.

        Creates an Adam optimizer for all decoder parameters with the specified
        learning rate.

        Returns:
            Optimizer: Adam optimizer instance configured for the decoders.
        """
        decoders_parameters = []
        for decoder_idx in range(len(self.model.pool_layers)):
            decoders_parameters.extend(list(self.model.decoders[decoder_idx].parameters()))

        return optim.Adam(
            params=decoders_parameters,
            lr=self.learning_rate,
        )

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a training step of the CFLOW model.

        The training process involves:
        1. Extract features using the encoder
        2. Process features in fiber batches
        3. Apply positional encoding
        4. Train decoders using normalizing flows

        Args:
            batch (Batch): Input batch containing images
            *args: Additional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Dictionary containing the average loss for the batch

        Raises:
            ValueError: If the fiber batch size is too large for the input size
        """
        del args, kwargs  # These variables are not used.

        opt = self.optimizers()

        images: torch.Tensor = batch.image
        activation = self.model.encoder(images)
        avg_loss = torch.zeros([1], dtype=torch.float64).to(images.device)

        height = []
        width = []
        for layer_idx, layer in enumerate(self.model.pool_layers):
            encoder_activations = activation[layer].detach()  # BxCxHxW

            batch_size, dim_feature_vector, im_height, im_width = encoder_activations.size()
            image_size = im_height * im_width
            embedding_length = batch_size * image_size  # number of rows in the conditional vector

            height.append(im_height)
            width.append(im_width)
            # repeats positional encoding for the entire batch 1 C H W to B C H W
            pos_encoding = einops.repeat(
                positional_encoding_2d(self.model.condition_vector, im_height, im_width).unsqueeze(0),
                "b c h w-> (tile b) c h w",
                tile=batch_size,
            ).to(images.device)
            c_r = einops.rearrange(pos_encoding, "b c h w -> (b h w) c")  # BHWxP
            e_r = einops.rearrange(encoder_activations, "b c h w -> (b h w) c")  # BHWxC
            perm = torch.randperm(embedding_length)  # BHW
            decoder = self.model.decoders[layer_idx].to(images.device)

            fiber_batches = embedding_length // self.model.fiber_batch_size  # number of fiber batches
            if fiber_batches <= 0:
                msg = "Make sure we have enough fibers, otherwise decrease N or batch-size!"
                raise ValueError(msg)

            for batch_num in range(fiber_batches):  # per-fiber processing
                opt.zero_grad()
                if batch_num < (fiber_batches - 1):
                    idx = torch.arange(
                        batch_num * self.model.fiber_batch_size,
                        (batch_num + 1) * self.model.fiber_batch_size,
                    )
                else:  # When non-full batch is encountered batch_num * N will go out of bounds
                    idx = torch.arange(batch_num * self.model.fiber_batch_size, embedding_length)
                # get random vectors
                c_p = c_r[perm[idx]]  # NxP
                e_p = e_r[perm[idx]]  # NxC
                # decoder returns the transformed variable z and the log Jacobian determinant
                p_u, log_jac_det = decoder(e_p, [c_p])
                decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
                log_prob = decoder_log_prob / dim_feature_vector  # likelihood per dim
                loss = -F.logsigmoid(log_prob)
                self.manual_backward(loss.mean())
                opt.step()
                avg_loss += loss.sum()

        self.log("train_loss", avg_loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": avg_loss}
    

    def _compute_loss(
        self,
        batch: Batch,
        per_image: bool = False,
        ) -> torch.Tensor | None:

        """Compute CFlow normalizing flow loss. 
            Args:
                batch: Input batch.
                per_image: If True, returns loss per image [B] using all images (test).
                       If False, returns scalar loss on normal images only (validation).
        """
        if per_image:
            images = batch.image  # all images — [B, C, H, W]
        else:
            normal_mask = batch.gt_label == 0
            if normal_mask.sum() == 0:
                return None
            images = batch.image[normal_mask]

        self.model.train()
        with torch.no_grad():

            if per_image:
                # process each image individually to get per-image loss
                losses = []
                for i in range(images.shape[0]):
                    single_image = images[i].unsqueeze(0)  # [1, C, H, W]
                    img_loss = self._compute_cflow_loss_scalar(single_image)
                    losses.append(img_loss)
                self.model.eval()
                return torch.stack(losses)  # [B]

            else:
                scalar = self._compute_cflow_loss_scalar(images)
                self.model.eval()
                return scalar


    def _compute_cflow_loss_scalar(self, images: torch.Tensor) -> torch.Tensor:
        """Run the fiber batch loop on given images and return a normalised scalar."""
        activation = self.model.encoder(images)
        avg_loss = torch.zeros([1], dtype=torch.float64).to(images.device)
        num_layers = len(self.model.pool_layers)

        for layer_idx, layer in enumerate(self.model.pool_layers):
            encoder_activations = activation[layer].detach()
            batch_size, dim_feature_vector, im_height, im_width = encoder_activations.size()
            image_size = im_height * im_width
            embedding_length = batch_size * image_size

            pos_encoding = einops.repeat(
                positional_encoding_2d(self.model.condition_vector, im_height, im_width).unsqueeze(0),
                "b c h w-> (tile b) c h w",
                tile=batch_size,
            ).to(images.device)
            c_r = einops.rearrange(pos_encoding, "b c h w -> (b h w) c")
            e_r = einops.rearrange(encoder_activations, "b c h w -> (b h w) c")
            perm = torch.randperm(embedding_length)
            decoder = self.model.decoders[layer_idx].to(images.device)

            fiber_batches = embedding_length // self.model.fiber_batch_size
            if fiber_batches <= 0:
                return torch.tensor(0.0).to(images.device)

            layer_loss = torch.zeros([1], dtype=torch.float64).to(images.device)
            for batch_num in range(fiber_batches):
                if batch_num < (fiber_batches - 1):
                    idx = torch.arange(
                        batch_num * self.model.fiber_batch_size,
                        (batch_num + 1) * self.model.fiber_batch_size,
                    )
                else:
                    idx = torch.arange(batch_num * self.model.fiber_batch_size, embedding_length)

                c_p = c_r[perm[idx]]
                e_p = e_r[perm[idx]]
                p_u, log_jac_det = decoder(e_p, [c_p])
                decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
                log_prob = decoder_log_prob / dim_feature_vector
                loss = -F.logsigmoid(log_prob)
                layer_loss += loss.mean()

            avg_loss += layer_loss / fiber_batches
        return (avg_loss / num_layers).float()



    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs

        val_loss = self._compute_loss(batch, per_image=False)
        if val_loss is not None:
            self.log("val_loss", val_loss.item(), on_epoch=True, prog_bar=True, logger=True)

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())



    def test_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs, batch_idx

        self._test_loss_cache = self._compute_loss(batch, per_image=True)

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get CFLOW-specific trainer arguments.

        Returns:
            dict[str, Any]: Dictionary containing trainer arguments:
                - gradient_clip_val: 0
                - num_sanity_val_steps: 0
        """
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type of the model.

        Returns:
            LearningType: ONE_CLASS learning type
        """
        return LearningType.ONE_CLASS
