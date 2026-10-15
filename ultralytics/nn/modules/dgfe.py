"""Difference-map guided feature enhancement for RT-DETR."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "DifferenceMapGuidedEnhancement",
    "P2DifferenceMapEnhancement",
    "OWRTDMDGFE",
    "PaperOWRTDMDGFE",
    "PaperOWRTDMDGFERHFree",
    "PaperOWRTDMDGFELiteRH320",
    "PaperOWRTDMDGFERH320",
    "P2PrimeGuidedP3Enhancement",
    "P2SemanticGuidedDifferenceMapEnhancement",
    "DifferenceMapGuidedCrossScaleFusion",
    "DifferenceMapGuidedDualMaskFusion",
    "LocalizationAwareDualMaskFusion",
    "P2GuidedDifferenceMapEnhancement",
)


class ReconstructionUpBlock(nn.Module):
    """Double-resolution reconstruction block used by the SR-TOD head."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PaperReconstructionUpBlock(nn.Module):
    """Reconstruction block matching the released SR-TOD implementation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels // 2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DifferenceMapGuidedEnhancement(nn.Module):
    """SR-TOD-style reconstruction, difference map and DGFE for a P3 feature.

    The standard RT-DETR-R50 neck exposes its highest-resolution decoder input
    at stride 8. Three reconstruction blocks therefore restore it to the input
    resolution. The forward binary mask uses a straight-through sigmoid
    surrogate so that the learnable threshold receives gradients.
    """

    def __init__(
        self,
        channels: int,
        recon_weight: float = 0.1,
        threshold_init: float = 0.0156862,
        threshold_beta: float = 50.0,
        reduction: int = 16,
        threshold_mode: str = "learnable",
        warmup_epochs: int = 0,
        adaptive_quantile: float = 0.95,
        fusion_mode: str = "legacy",
        alpha: float = 1.0,
        mask_resize_mode: str = "nearest",
        detach_mask: bool = False,
        reconstruction_train_epochs: int = -1,
    ):
        super().__init__()
        if threshold_mode not in {"fixed", "learnable", "adaptive_quantile"}:
            raise ValueError(
                "threshold_mode must be 'fixed', 'learnable' or 'adaptive_quantile', "
                f"got {threshold_mode!r}"
            )
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be non-negative, got {warmup_epochs}")
        if not 0.0 < adaptive_quantile < 1.0:
            raise ValueError(f"adaptive_quantile must be between 0 and 1, got {adaptive_quantile}")
        if fusion_mode not in {"legacy", "residual", "gated_residual"}:
            raise ValueError(
                "fusion_mode must be 'legacy', 'residual' or 'gated_residual', "
                f"got {fusion_mode!r}"
            )
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if mask_resize_mode not in {"nearest", "area", "max"}:
            raise ValueError(
                "mask_resize_mode must be 'nearest', 'area' or 'max', "
                f"got {mask_resize_mode!r}"
            )

        c1 = max(channels // 2, 64)
        c2 = max(c1 // 2, 32)
        c3 = max(c2 // 2, 16)
        hidden = max(channels // reduction, 4)

        self.reconstruction_head = nn.Sequential(
            ReconstructionUpBlock(channels, c1),
            ReconstructionUpBlock(c1, c2),
            ReconstructionUpBlock(c2, c3),
            nn.Conv2d(c3, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        threshold = torch.tensor(float(threshold_init))
        if threshold_mode == "learnable":
            self.threshold = nn.Parameter(threshold)
        else:
            self.register_buffer("threshold", threshold)
        # ModelEMA in this Ultralytics fork only updates floating-point state,
        # so keep the epoch as a float buffer to make validation/checkpoints
        # follow the same warm-up phase as the training model.
        self.register_buffer("_current_epoch", torch.tensor(0.0))
        self.recon_weight = float(recon_weight)
        self.threshold_beta = float(threshold_beta)
        self.threshold_mode = threshold_mode
        self.warmup_epochs = int(warmup_epochs)
        self.adaptive_quantile = float(adaptive_quantile)
        self.fusion_mode = fusion_mode
        self.alpha = float(alpha)
        self.mask_resize_mode = mask_resize_mode
        self.detach_mask = bool(detach_mask)
        self.reconstruction_train_epochs = int(reconstruction_train_epochs)

        self.reconstruction_loss: torch.Tensor | None = None
        self.last_input_image: torch.Tensor | None = None
        self.last_reconstructed_image: torch.Tensor | None = None
        self.last_difference_map: torch.Tensor | None = None
        self.last_binary_map: torch.Tensor | None = None
        self.last_feature_mask: torch.Tensor | None = None
        self.last_p2_feature_mask: torch.Tensor | None = None
        self.last_gated_residual: torch.Tensor | None = None
        self.last_channel_attention: torch.Tensor | None = None
        self.last_effective_threshold: torch.Tensor | None = None
        self.last_feature_relative_change: torch.Tensor | None = None

    @property
    def current_epoch(self) -> int:
        """Current zero-based training epoch used by the DGFE warm-up schedule."""
        return int(self._current_epoch.item())

    @property
    def enhancement_active(self) -> bool:
        """Whether Difference Map Guided Feature Enhancement is currently enabled."""
        return self.current_epoch >= self.warmup_epochs

    @property
    def reconstruction_training_active(self) -> bool:
        """Whether the reconstruction head should still receive gradients."""
        train_epochs = int(getattr(self, "reconstruction_train_epochs", -1))
        return train_epochs < 0 or self.current_epoch < train_epochs

    def set_epoch(self, epoch: int) -> None:
        """Update the zero-based epoch without replacing the persistent EMA buffer."""
        self._current_epoch.fill_(float(epoch))

    def __getstate__(self):
        """Exclude per-forward tensors when trainers deepcopy the model for EMA."""
        state = self.__dict__.copy()
        for name in (
            "reconstruction_loss",
            "last_input_image",
            "last_reconstructed_image",
            "last_difference_map",
            "last_predicted_difference_map",
            "last_predicted_filtration_map",
            "last_binary_map",
            "last_feature_mask",
            "last_p2_feature_mask",
            "last_gated_residual",
            "last_semantic_gate",
            "last_final_mask",
            "last_p2_guidance",
            "last_fused_guidance",
            "last_cross_scale_residual",
            "last_p95_mask",
            "last_p90_mask",
            "last_mask_gate",
            "last_p90_effective_threshold",
            "last_difference_map_p2",
            "last_difference_map_p2_norm",
            "last_high_frequency_difference_map",
            "last_attention_matrix",
            "last_filtration_map",
            "last_channel_reweight",
            "last_p2_input",
            "last_p2_enhanced",
            "last_p2_residual",
            "dm_supervision_mask",
            "last_localization_gate",
            "last_p2_local_detail",
            "last_localization_residual",
            "last_channel_attention",
            "last_effective_threshold",
            "last_feature_relative_change",
        ):
            state[name] = None
        return state

    def resize_mask(
        self,
        binary_mask: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Project an input-resolution Difference Map onto a feature grid."""
        mask = binary_mask.detach() if getattr(self, "detach_mask", False) else binary_mask
        mode = getattr(self, "mask_resize_mode", "nearest")
        if mode == "max":
            if (
                target_size[0] <= mask.shape[-2]
                and target_size[1] <= mask.shape[-1]
            ):
                return F.adaptive_max_pool2d(mask, target_size)
            return F.interpolate(mask, size=target_size, mode="nearest")
        if mode == "area":
            return F.interpolate(mask, size=target_size, mode="area")
        return F.interpolate(mask, size=target_size, mode="nearest")

    def forward(self, feature: torch.Tensor, input_image: torch.Tensor) -> torch.Tensor:
        if self.reconstruction_training_active:
            reconstructed = self.reconstruction_head(feature)
        else:
            with torch.no_grad():
                reconstructed = self.reconstruction_head(feature)
        if reconstructed.shape[-2:] != input_image.shape[-2:]:
            reconstructed = F.interpolate(
                reconstructed, size=input_image.shape[-2:], mode="bilinear", align_corners=False
            )

        target = input_image.float()
        reconstructed_f32 = reconstructed.float()
        self.reconstruction_loss = (
            F.mse_loss(reconstructed_f32, target)
            if self.reconstruction_training_active
            else target.new_zeros(())
        )

        difference = torch.abs(reconstructed_f32 - target).mean(dim=1, keepdim=True)
        if self.threshold_mode == "adaptive_quantile":
            # A detached per-image threshold makes the top (1-q) response
            # sparse by construction while preserving gradients through the
            # soft mask to the reconstruction branch.
            effective_threshold = torch.quantile(
                difference.detach().flatten(1),
                self.adaptive_quantile,
                dim=1,
            ).view(-1, 1, 1, 1)
        else:
            effective_threshold = self.threshold.view(1, 1, 1, 1)
        soft_mask = torch.sigmoid(self.threshold_beta * (difference - effective_threshold))
        hard_mask = (difference >= effective_threshold).to(soft_mask.dtype)
        binary_mask = hard_mask.detach() - soft_mask.detach() + soft_mask
        feature_mask = self.resize_mask(binary_mask, feature.shape[-2:])

        avg_context = F.adaptive_avg_pool2d(feature, output_size=1)
        max_context = F.adaptive_max_pool2d(feature, output_size=1)
        channel_attention = torch.sigmoid(self.channel_mlp(avg_context) + self.channel_mlp(max_context))

        # Keep the legacy paper-aligned path for old configurations and
        # checkpoints. The controlled residual path leaves background features
        # unchanged and only applies a bounded positive gain inside the sparse
        # Difference Map.
        fusion_mode = getattr(self, "fusion_mode", "legacy")
        alpha = float(getattr(self, "alpha", 1.0))
        if fusion_mode == "residual":
            enhanced = feature * (1.0 + alpha * channel_attention * feature_mask)
        else:
            enhanced = feature * channel_attention
            enhanced = enhanced * (1.0 + feature_mask)

        self.last_input_image = target.detach()
        self.last_reconstructed_image = reconstructed_f32.detach()
        self.last_difference_map = difference.detach()
        self.last_binary_map = binary_mask.detach()
        self.last_feature_mask = feature_mask.detach()
        self.last_channel_attention = channel_attention.detach()
        self.last_effective_threshold = effective_threshold.detach()
        # Stable strategy: reconstruction is optimized during warm-up, while
        # detection sees the unmodified feature until the scheduled epoch.
        output = enhanced if self.enhancement_active else feature
        self.last_feature_relative_change = (
            (output.detach() - feature.detach()).abs().mean()
            / feature.detach().abs().mean().clamp_min(1e-12)
        )
        return output


class P2DifferenceMapEnhancement(DifferenceMapGuidedEnhancement):
    """Paper-aligned Difference Map enhancement applied directly to P2.

    The original SR-TOD/DGFE formulation reconstructs the input image from the
    stride-4 P2 feature with two upsampling blocks, then replaces P2 with the
    enhanced feature P2'. Placing this module immediately after the ResNet P2
    stage lets the enhanced feature continue through P3-P5 and the RT-DETR
    encoder without introducing an OWRT-specific neck dependency.
    """

    def __init__(
        self,
        channels: int,
        recon_weight: float = 0.1,
        threshold_init: float = 0.0156862,
        threshold_beta: float = 50.0,
        reduction: int = 16,
        threshold_mode: str = "learnable",
        warmup_epochs: int = 0,
        adaptive_quantile: float = 0.95,
        fusion_mode: str = "legacy",
        alpha: float = 1.0,
        mask_resize_mode: str = "nearest",
        detach_mask: bool = False,
        reconstruction_train_epochs: int = -1,
    ):
        super().__init__(
            channels=channels,
            recon_weight=recon_weight,
            threshold_init=threshold_init,
            threshold_beta=threshold_beta,
            reduction=reduction,
            threshold_mode=threshold_mode,
            warmup_epochs=warmup_epochs,
            adaptive_quantile=adaptive_quantile,
            fusion_mode=fusion_mode,
            alpha=alpha,
            mask_resize_mode=mask_resize_mode,
            detach_mask=detach_mask,
            reconstruction_train_epochs=reconstruction_train_epochs,
        )
        c1 = max(channels // 2, 64)
        c2 = max(c1 // 2, 32)
        self.reconstruction_head = nn.Sequential(
            PaperReconstructionUpBlock(channels, c1),
            PaperReconstructionUpBlock(c1, c2),
            nn.Conv2d(c2, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )


class OWRTDMDGFE(DifferenceMapGuidedEnhancement):
    """SR-TOD-style P2 -> P2' DGFE adapted for the OWRT-DETR neck.

    This module follows the Difference Map paper more closely than the earlier
    OWRT-DM/M4 branch: P2 is reconstructed to the input image, a pixel-level
    Difference Map is projected back to the P2 grid, and a DGFE attention matrix
    enhances P2 itself. The output has the same shape as P2 and can therefore
    replace P2 in OWRT-DETR's CFPIM/MSSF path without changing the original
    decoder API.
    """

    def __init__(
        self,
        channels: int,
        recon_weight: float = 0.1,
        alpha: float = 1.0,
        beta: float = 0.5,
        high_quantile: float = 0.95,
        low_quantile: float = 0.90,
        threshold_beta: float = 50.0,
        reduction: int = 16,
        mask_resize_mode: str = "bilinear",
        detach_mask: bool = False,
        reconstruction_train_epochs: int = -1,
    ):
        super().__init__(
            channels=channels,
            recon_weight=recon_weight,
            threshold_init=0.0156862,
            threshold_beta=threshold_beta,
            reduction=reduction,
            threshold_mode="adaptive_quantile",
            warmup_epochs=0,
            adaptive_quantile=high_quantile,
            fusion_mode="residual",
            alpha=alpha,
            mask_resize_mode="nearest",
            detach_mask=detach_mask,
            reconstruction_train_epochs=reconstruction_train_epochs,
        )
        if not 0.0 < low_quantile < high_quantile < 1.0:
            raise ValueError(
                "Expected 0 < low_quantile < high_quantile < 1, "
                f"got low={low_quantile}, high={high_quantile}"
            )
        if mask_resize_mode not in {"nearest", "bilinear", "area"}:
            raise ValueError(
                "mask_resize_mode must be 'nearest', 'bilinear' or 'area', "
                f"got {mask_resize_mode!r}"
            )
        c1 = max(channels // 2, 64)
        c2 = max(c1 // 2, 32)
        hidden = max(channels // reduction, 16)

        # P2 is stride-4, so two upsampling blocks recover the input resolution.
        self.reconstruction_head = nn.Sequential(
            PaperReconstructionUpBlock(channels, c1),
            PaperReconstructionUpBlock(c1, c2),
            nn.Conv2d(c2, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.mask_to_attention = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.residual_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )
        # Start as a near-identity adapter for stable transfer from OWRT weights.
        nn.init.zeros_(self.residual_conv[-1].weight)
        nn.init.zeros_(self.residual_conv[-1].bias)

        self.beta = float(beta)
        self.high_quantile = float(high_quantile)
        self.low_quantile = float(low_quantile)
        self.mask_resize_mode = mask_resize_mode

        self.last_p95_mask: torch.Tensor | None = None
        self.last_p90_mask: torch.Tensor | None = None
        self.last_final_mask: torch.Tensor | None = None
        self.last_difference_map_p2: torch.Tensor | None = None
        self.last_difference_map_p2_norm: torch.Tensor | None = None
        self.last_high_frequency_difference_map: torch.Tensor | None = None
        self.last_attention_matrix: torch.Tensor | None = None
        self.last_p2_input: torch.Tensor | None = None
        self.last_p2_enhanced: torch.Tensor | None = None
        self.last_p2_residual: torch.Tensor | None = None

    @staticmethod
    def _normalize_per_image(x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        flat = x.flatten(1)
        min_v = flat.min(dim=1).values.view(b, 1, 1, 1)
        max_v = flat.max(dim=1).values.view(b, 1, 1, 1)
        return (x - min_v) / (max_v - min_v).clamp_min(1e-6)

    def _resize_difference_to_p2(self, difference: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        mode = getattr(self, "mask_resize_mode", "bilinear")
        if mode == "area":
            return F.interpolate(difference, size=size, mode="area")
        if mode == "nearest":
            return F.interpolate(difference, size=size, mode="nearest")
        return F.interpolate(difference, size=size, mode="bilinear", align_corners=False)

    def forward(self, p2_feature: torch.Tensor, input_image: torch.Tensor) -> torch.Tensor:
        if self.reconstruction_training_active:
            reconstructed = self.reconstruction_head(p2_feature)
        else:
            with torch.no_grad():
                reconstructed = self.reconstruction_head(p2_feature)
        if reconstructed.shape[-2:] != input_image.shape[-2:]:
            reconstructed = F.interpolate(
                reconstructed,
                size=input_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        target = input_image.float()
        reconstructed_f32 = reconstructed.float()
        self.reconstruction_loss = (
            F.mse_loss(reconstructed_f32, target)
            if self.reconstruction_training_active
            else target.new_zeros(())
        )

        difference = torch.abs(reconstructed_f32 - target).mean(dim=1, keepdim=True)
        difference_p2 = self._resize_difference_to_p2(difference, p2_feature.shape[-2:])
        difference_norm = self._normalize_per_image(difference_p2)

        high_threshold = torch.quantile(
            difference_norm.detach().flatten(1),
            self.high_quantile,
            dim=1,
        ).view(-1, 1, 1, 1)
        low_threshold = torch.quantile(
            difference_norm.detach().flatten(1),
            self.low_quantile,
            dim=1,
        ).view(-1, 1, 1, 1)
        high_soft = torch.sigmoid(self.threshold_beta * (difference_norm - high_threshold))
        high_hard = (difference_norm >= high_threshold).to(high_soft.dtype)
        p95_hard = high_hard.detach() - high_soft.detach() + high_soft
        p90_soft = torch.sigmoid(self.threshold_beta * (difference_norm - low_threshold))

        final_mask = (p95_hard + self.beta * p90_soft).clamp(0.0, 1.0)
        if getattr(self, "detach_mask", False):
            final_mask_for_attention = final_mask.detach()
        else:
            final_mask_for_attention = final_mask

        attention = self.mask_to_attention(final_mask_for_attention)
        residual = self.residual_conv(p2_feature * attention)
        alpha = float(getattr(self, "alpha", 1.0))
        enhanced = p2_feature + alpha * residual
        output = enhanced if self.enhancement_active else p2_feature

        # A lightweight high-frequency map for paper-style visualization.
        smooth = F.avg_pool2d(difference_norm, kernel_size=5, stride=1, padding=2)
        high_frequency = torch.abs(difference_norm - smooth)

        self.last_input_image = target.detach()
        self.last_reconstructed_image = reconstructed_f32.detach()
        self.last_difference_map = difference.detach()
        self.last_difference_map_p2 = difference_p2.detach()
        self.last_difference_map_p2_norm = difference_norm.detach()
        self.last_high_frequency_difference_map = high_frequency.detach()
        self.last_binary_map = p95_hard.detach()
        self.last_p95_mask = p95_hard.detach()
        self.last_p90_mask = p90_soft.detach()
        self.last_feature_mask = final_mask.detach()
        self.last_final_mask = final_mask.detach()
        self.last_attention_matrix = attention.detach()
        self.last_p2_input = p2_feature.detach()
        self.last_p2_enhanced = output.detach()
        self.last_p2_residual = (output.detach() - p2_feature.detach())
        self.last_effective_threshold = high_threshold.detach()
        self.last_p90_effective_threshold = low_threshold.detach()
        self.last_feature_relative_change = (
            (output.detach() - p2_feature.detach()).abs().mean()
            / p2_feature.detach().abs().mean().clamp_min(1e-12)
        )
        return output


class PaperOWRTDMDGFE(DifferenceMapGuidedEnhancement):
    """Paper-faithful SR-TOD DGFE for OWRT-DETR.

    This is the strict comparison module for the improved ``OWRTDMDGFE``. It
    follows the SR-TOD paper formulation:

        D = MeanChannel(|Ir - Io|)
        Db = (Sign(D - t) + 1) * 0.5
        Filtration(D) = Resize(Db) + 1
        Reweighting(P2) = sigmoid(MLP(AvgPool(P2)) + MLP(MaxPool(P2)))
        P2' = (Reweighting(P2) * Filtration(D)) * P2

    It deliberately does not use P95/P90 dual thresholds, final masks, residual
    convolutions, alpha gains, or ``P2 + Conv(P2*M)`` style enhancement.
    """

    def __init__(
        self,
        channels: int,
        recon_weight: float = 0.1,
        threshold_init: float = 0.0156862,
        threshold_beta: float = 50.0,
        reduction: int = 16,
        paper_strict: bool = True,
        reconstruction_train_epochs: int = -1,
        threshold_mode: str = "learnable",
        adaptive_quantile: float = 0.95,
    ):
        super().__init__(
            channels=channels,
            recon_weight=recon_weight,
            threshold_init=threshold_init,
            threshold_beta=threshold_beta,
            reduction=reduction,
            threshold_mode=threshold_mode,
            warmup_epochs=0,
            adaptive_quantile=adaptive_quantile,
            fusion_mode="legacy",
            alpha=1.0,
            mask_resize_mode="nearest",
            detach_mask=False,
            reconstruction_train_epochs=reconstruction_train_epochs,
        )
        c1 = max(channels // 2, 64)
        c2 = max(c1 // 2, 32)
        self.reconstruction_head = nn.Sequential(
            PaperReconstructionUpBlock(channels, c1),
            PaperReconstructionUpBlock(c1, c2),
            nn.Conv2d(c2, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.paper_strict = bool(paper_strict)
        self.last_filtration_map: torch.Tensor | None = None
        self.last_channel_reweight: torch.Tensor | None = None
        self.last_attention_matrix: torch.Tensor | None = None
        self.last_p2_input: torch.Tensor | None = None
        self.last_p2_enhanced: torch.Tensor | None = None
        self.last_p2_residual: torch.Tensor | None = None

    def forward(self, p2_feature: torch.Tensor, input_image: torch.Tensor) -> torch.Tensor:
        if self.reconstruction_training_active:
            reconstructed = self.reconstruction_head(p2_feature)
        else:
            with torch.no_grad():
                reconstructed = self.reconstruction_head(p2_feature)
        if reconstructed.shape[-2:] != input_image.shape[-2:]:
            reconstructed = F.interpolate(
                reconstructed,
                size=input_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # In this Ultralytics pipeline batch['img'] is already 0-1 float. Keep
        # the paper branch in the same 0-1 space as the sigmoid reconstruction.
        raw_img_01 = input_image.float()
        reconstructed_f32 = reconstructed.float()
        self.reconstruction_loss = (
            F.mse_loss(reconstructed_f32, raw_img_01)
            if self.reconstruction_training_active
            else raw_img_01.new_zeros(())
        )

        difference = torch.abs(reconstructed_f32 - raw_img_01).mean(dim=1, keepdim=True)
        if getattr(self, "threshold_mode", "learnable") == "adaptive_quantile":
            condition_q = getattr(self, "condition_quantiles", None)
            if condition_q is None:
                threshold = torch.quantile(
                    difference.detach().flatten(1), float(self.adaptive_quantile), dim=1,
                ).view(-1, 1, 1, 1)
            else:
                if len(condition_q) != len(difference):
                    raise ValueError("One teacher quantile is required per input image")
                threshold = torch.stack([
                    torch.quantile(row, float(q))
                    for row, q in zip(difference.detach().flatten(1), condition_q)
                ]).view(-1, 1, 1, 1)
        else:
            threshold = self.threshold.view(1, 1, 1, 1)
        threshold = threshold.to(device=difference.device, dtype=difference.dtype)
        if self.paper_strict:
            db = (difference >= threshold).to(difference.dtype)
        else:
            soft = torch.sigmoid(float(getattr(self, "threshold_beta", 50.0)) * (difference - threshold))
            hard = (difference >= threshold).to(soft.dtype)
            db = hard.detach() - soft.detach() + soft

        db_p2 = F.interpolate(db, size=p2_feature.shape[-2:], mode="nearest")
        filtration = db_p2 + 1.0

        kernel_size = p2_feature.shape[-2:]
        avg_context = F.avg_pool2d(p2_feature, kernel_size=kernel_size)
        max_context = F.max_pool2d(p2_feature, kernel_size=kernel_size)
        channel_reweight = torch.sigmoid(self.channel_mlp(avg_context) + self.channel_mlp(max_context))

        attention_matrix = channel_reweight * filtration
        p2_prime = p2_feature * attention_matrix
        output = p2_prime if self.enhancement_active else p2_feature

        self.last_input_image = raw_img_01.detach()
        self.last_reconstructed_image = reconstructed_f32.detach()
        self.last_difference_map = difference.detach()
        self.last_binary_map = db.detach()
        self.last_feature_mask = filtration.detach()
        self.last_filtration_map = filtration.detach()
        self.last_channel_reweight = channel_reweight.detach()
        self.last_attention_matrix = attention_matrix.detach()
        self.last_p2_input = p2_feature.detach()
        self.last_p2_enhanced = output.detach()
        self.last_p2_residual = (output.detach() - p2_feature.detach())
        self.last_effective_threshold = threshold.detach()
        self.last_feature_relative_change = (
            (output.detach() - p2_feature.detach()).abs().mean()
            / p2_feature.detach().abs().mean().clamp_min(1e-12)
        )
        return output


class PaperOWRTDMDGFERHFree(PaperOWRTDMDGFE):
    """RH-free adaptive DGFE with a lightweight P2 mask head.

    This module keeps the paper DGFE output contract (P2 -> P2') but removes the
    expensive reconstruction head. A small depthwise-separable mask head predicts
    a soft Difference Map and Filtration Map directly from P2. During KD, these
    maps can be supervised by a full adaptive teacher that still owns RH.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        mask_hidden: int = 64,
        mask_scale: float = 1.0,
        detach_mask: bool = False,
    ):
        super().__init__(
            channels=channels,
            recon_weight=0.0,
            threshold_init=0.0156862,
            threshold_beta=50.0,
            reduction=reduction,
            paper_strict=False,
            reconstruction_train_epochs=0,
            threshold_mode="fixed",
            adaptive_quantile=0.97,
        )
        hidden = max(int(mask_hidden), max(channels // 4, 32))
        self.reconstruction_head = nn.Identity()
        self.mask_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1),
        )
        self.recon_weight = 0.0
        self.mask_scale = float(mask_scale)
        self.detach_mask = bool(detach_mask)
        self.last_predicted_difference_map: torch.Tensor | None = None
        self.last_predicted_filtration_map: torch.Tensor | None = None

    def forward(self, p2_feature: torch.Tensor, input_image: torch.Tensor) -> torch.Tensor:
        self.reconstruction_loss = p2_feature.new_zeros(())

        map_logits = self.mask_head(p2_feature)
        scale = max(float(self.mask_scale), 1e-6)
        difference_soft = torch.sigmoid(map_logits[:, :1] * scale)
        filtration_soft = torch.sigmoid(map_logits[:, 1:2] * scale)
        if self.detach_mask:
            difference_for_gate = difference_soft.detach()
            filtration_for_gate = filtration_soft.detach()
        else:
            difference_for_gate = difference_soft
            filtration_for_gate = filtration_soft

        filtration = 1.0 + filtration_for_gate
        kernel_size = p2_feature.shape[-2:]
        avg_context = F.avg_pool2d(p2_feature, kernel_size=kernel_size)
        max_context = F.max_pool2d(p2_feature, kernel_size=kernel_size)
        channel_reweight = torch.sigmoid(self.channel_mlp(avg_context) + self.channel_mlp(max_context))

        attention_matrix = channel_reweight * filtration
        p2_prime = p2_feature * attention_matrix
        output = p2_prime if self.enhancement_active else p2_feature

        self.last_input_image = input_image.detach() if torch.is_tensor(input_image) else None
        self.last_reconstructed_image = None
        self.last_difference_map = difference_soft.detach()
        self.last_difference_map_p2 = difference_soft.detach()
        self.last_difference_map_p2_norm = difference_soft.detach()
        self.last_binary_map = filtration_soft.detach()
        self.last_feature_mask = (1.0 + filtration_soft).detach()
        self.last_filtration_map = (1.0 + filtration_soft).detach()
        # Distillation needs the graph; visualization caches above remain detached.
        self.last_predicted_difference_map = difference_soft if self.training else difference_soft.detach()
        self.last_predicted_filtration_map = filtration_soft if self.training else filtration_soft.detach()
        self.last_channel_reweight = channel_reweight.detach()
        self.last_attention_matrix = attention_matrix.detach()
        self.last_p2_input = p2_feature.detach()
        self.last_p2_enhanced = output.detach()
        self.last_p2_residual = output.detach() - p2_feature.detach()
        self.last_effective_threshold = p2_feature.new_zeros((p2_feature.shape[0], 1, 1, 1)).detach()
        self.last_feature_relative_change = (
            (output.detach() - p2_feature.detach()).abs().mean()
            / p2_feature.detach().abs().mean().clamp_min(1e-12)
        )
        return output


class PaperOWRTDMDGFERH320(PaperOWRTDMDGFE):
    """Lightweight Paper-adaptive DGFE with half-resolution reconstruction.

    Compared with PaperOWRTDMDGFE, this version keeps the same Difference Map,
    adaptive quantile filtration, channel reweighting and P2' formula, but the
    reconstruction head only upsamples P2 once. The reconstructed image is
    generated at 320x320 for 640x640 input, and the raw image is downsampled to
    the same size before computing the Difference Map.

    Main purpose:
        reduce RH GFLOPs while preserving the Paper-DGFE logic.
    """

    def __init__(
        self,
        channels: int,
        recon_weight: float = 0.1,
        threshold_init: float = 0.0156862,
        threshold_beta: float = 50.0,
        reduction: int = 16,
        paper_strict: bool = True,
        reconstruction_train_epochs: int = -1,
        threshold_mode: str = "adaptive_quantile",
        adaptive_quantile: float = 0.97,
    ):
        super().__init__(
            channels=channels,
            recon_weight=recon_weight,
            threshold_init=threshold_init,
            threshold_beta=threshold_beta,
            reduction=reduction,
            paper_strict=paper_strict,
            reconstruction_train_epochs=reconstruction_train_epochs,
            threshold_mode=threshold_mode,
            adaptive_quantile=adaptive_quantile,
        )
        c1 = max(channels // 2, 64)
        self.reconstruction_head = nn.Sequential(
            PaperReconstructionUpBlock(channels, c1),
            nn.Conv2d(c1, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, p2_feature: torch.Tensor, input_image: torch.Tensor) -> torch.Tensor:
        if self.reconstruction_training_active:
            reconstructed = self.reconstruction_head(p2_feature)
        else:
            with torch.no_grad():
                reconstructed = self.reconstruction_head(p2_feature)

        raw_img_01 = input_image.float()
        reconstructed_f32 = reconstructed.float()

        if raw_img_01.shape[-2:] != reconstructed_f32.shape[-2:]:
            raw_for_diff = F.interpolate(
                raw_img_01,
                size=reconstructed_f32.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            raw_for_diff = raw_img_01

        self.reconstruction_loss = (
            F.mse_loss(reconstructed_f32, raw_for_diff)
            if self.reconstruction_training_active
            else raw_for_diff.new_zeros(())
        )

        difference = torch.abs(reconstructed_f32 - raw_for_diff).mean(dim=1, keepdim=True)

        if getattr(self, "threshold_mode", "learnable") == "adaptive_quantile":
            threshold = torch.quantile(
                difference.detach().flatten(1),
                float(getattr(self, "adaptive_quantile", 0.97)),
                dim=1,
            ).view(-1, 1, 1, 1)
        else:
            threshold = self.threshold.view(1, 1, 1, 1)

        threshold = threshold.to(device=difference.device, dtype=difference.dtype)

        if self.paper_strict:
            db = (torch.sign(difference - threshold) + 1.0) * 0.5
        else:
            soft = torch.sigmoid(float(getattr(self, "threshold_beta", 50.0)) * (difference - threshold))
            hard = (difference >= threshold).to(soft.dtype)
            db = hard.detach() - soft.detach() + soft

        db_p2 = F.interpolate(db, size=p2_feature.shape[-2:], mode="nearest")
        filtration = db_p2 + 1.0

        kernel_size = p2_feature.shape[-2:]
        avg_context = F.avg_pool2d(p2_feature, kernel_size=kernel_size)
        max_context = F.max_pool2d(p2_feature, kernel_size=kernel_size)
        channel_reweight = torch.sigmoid(self.channel_mlp(avg_context) + self.channel_mlp(max_context))

        attention_matrix = channel_reweight * filtration
        p2_prime = p2_feature * attention_matrix
        output = p2_prime if self.enhancement_active else p2_feature

        self.last_input_image = raw_for_diff.detach()
        self.last_reconstructed_image = reconstructed_f32.detach()
        self.last_difference_map = difference.detach()
        self.last_binary_map = db.detach()
        self.last_feature_mask = filtration.detach()
        self.last_filtration_map = filtration.detach()
        self.last_channel_reweight = channel_reweight.detach()
        self.last_attention_matrix = attention_matrix.detach()
        self.last_p2_input = p2_feature.detach()
        self.last_p2_enhanced = output.detach()
        self.last_p2_residual = (output.detach() - p2_feature.detach())
        self.last_effective_threshold = threshold.detach()
        self.last_feature_relative_change = (
            (output.detach() - p2_feature.detach()).abs().mean()
            / p2_feature.detach().abs().mean().clamp_min(1e-12)
        )
        return output



class P2PrimeGuidedP3Enhancement(nn.Module):
    """Inject an already-enhanced stride-4 P2' feature into stride-8 P3.

    This module is for a controlled ablation:

        Paper-adaptive DGFE: P2 -> P2'
        P2' -> P3 injection: P3 -> P3'
        Decoder inputs: P3', P4, P5'

    It intentionally does not build another reconstruction head or Difference
    Map. The goal is to isolate whether the PaperAdaptive P2' feature is better
    used as a direct fourth decoder scale or as detail guidance for P3.
    """

    def __init__(
        self,
        p2_prime_channels: int,
        target_channels: int,
        alpha: float = 1.0,
        mode: str = "gated_residual",
        resize_mode: str = "bilinear",
    ):
        super().__init__()
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if mode not in {"gated_residual", "residual_add", "multiplicative"}:
            raise ValueError(
                "mode must be 'gated_residual', 'residual_add' or 'multiplicative', "
                f"got {mode!r}"
            )
        if resize_mode not in {"nearest", "bilinear", "area"}:
            raise ValueError(
                "resize_mode must be 'nearest', 'bilinear' or 'area', "
                f"got {resize_mode!r}"
            )
        self.p2_prime_channels = int(p2_prime_channels)
        self.target_channels = int(target_channels)
        self.alpha = float(alpha)
        self.mode = str(mode)
        self.resize_mode = str(resize_mode)

        self.p2_to_p3 = nn.Sequential(
            nn.Conv2d(
                p2_prime_channels,
                p2_prime_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=p2_prime_channels,
                bias=False,
            ),
            nn.BatchNorm2d(p2_prime_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(p2_prime_channels, target_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(target_channels),
            nn.SiLU(inplace=True),
        )
        self.residual_gate = nn.Conv2d(target_channels, target_channels, kernel_size=1, bias=False)
        # Start from the transferred three-scale behavior. The gate learns how
        # much Paper-DGFE P2' detail should enter P3.
        nn.init.zeros_(self.residual_gate.weight)

        hidden = max(target_channels // 4, 32)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(target_channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.zeros_(self.spatial_gate[-1].bias)

        self.last_p2_prime: torch.Tensor | None = None
        self.last_p2_projected: torch.Tensor | None = None
        self.last_spatial_gate: torch.Tensor | None = None
        self.last_p3_before: torch.Tensor | None = None
        self.last_p3_after: torch.Tensor | None = None
        self.last_p3_residual: torch.Tensor | None = None
        self.last_feature_relative_change: torch.Tensor | None = None

    def _resize_to_target(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        mode = self.resize_mode
        if mode == "bilinear":
            return F.interpolate(x, size=size, mode=mode, align_corners=False)
        return F.interpolate(x, size=size, mode=mode)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) != 2:
            raise TypeError("P2PrimeGuidedP3Enhancement expects [P2_prime, target_P3]")
        p2_prime, target_feature = features
        p2_projected = self.p2_to_p3(p2_prime)
        p2_projected = self._resize_to_target(p2_projected, target_feature.shape[-2:])

        mode = getattr(self, "mode", "gated_residual")
        alpha = float(getattr(self, "alpha", 1.0))
        spatial_gate = 2.0 * torch.sigmoid(self.spatial_gate(target_feature))
        if mode == "gated_residual":
            residual = self.residual_gate(p2_projected * spatial_gate)
            enhanced = target_feature + alpha * residual
        elif mode == "residual_add":
            residual = p2_projected * spatial_gate
            enhanced = target_feature + alpha * residual
        else:
            residual = p2_projected * spatial_gate
            enhanced = target_feature * (1.0 + alpha * torch.sigmoid(residual))

        self.last_p2_prime = p2_prime.detach()
        self.last_p2_projected = p2_projected.detach()
        self.last_spatial_gate = spatial_gate.detach()
        self.last_p3_before = target_feature.detach()
        self.last_p3_after = enhanced.detach()
        self.last_p3_residual = (enhanced.detach() - target_feature.detach())
        self.last_feature_relative_change = (
            self.last_p3_residual.abs().mean()
            / target_feature.detach().abs().mean().clamp_min(1e-12)
        )
        return enhanced


class P2SemanticGuidedDifferenceMapEnhancement(P2DifferenceMapEnhancement):
    """P2 Difference Map enhancement gated by a higher-level semantic feature.

    This module receives ``[P2, P3_semantic]`` and returns an enhanced P2. The
    Difference Map still comes from P2 self-reconstruction, while a lightweight
    semantic gate from P3 modulates the sparse mask before residual fusion.
    The gate is initialized as identity via ``2 * sigmoid(0) = 1`` so the
    module starts from the paired M3-A behavior and can learn semantic filtering.
    """

    def __init__(
        self,
        p2_channels: int,
        semantic_channels: int,
        recon_weight: float = 1.0,
        threshold_init: float = 0.0156862,
        threshold_beta: float = 50.0,
        reduction: int = 16,
        threshold_mode: str = "adaptive_quantile",
        warmup_epochs: int = 0,
        adaptive_quantile: float = 0.95,
        fusion_mode: str = "residual",
        alpha: float = 0.1,
        mask_resize_mode: str = "nearest",
        detach_mask: bool = True,
        reconstruction_train_epochs: int = -1,
    ):
        super().__init__(
            channels=p2_channels,
            recon_weight=recon_weight,
            threshold_init=threshold_init,
            threshold_beta=threshold_beta,
            reduction=reduction,
            threshold_mode=threshold_mode,
            warmup_epochs=warmup_epochs,
            adaptive_quantile=adaptive_quantile,
            fusion_mode=fusion_mode,
            alpha=alpha,
            mask_resize_mode=mask_resize_mode,
            detach_mask=detach_mask,
            reconstruction_train_epochs=reconstruction_train_epochs,
        )
        hidden = max(semantic_channels // 4, 32)
        self.semantic_channels = int(semantic_channels)
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.semantic_gate[-1].weight)
        nn.init.zeros_(self.semantic_gate[-1].bias)
        self.last_semantic_gate: torch.Tensor | None = None
        self.last_final_mask: torch.Tensor | None = None

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        input_image: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) != 2:
            raise TypeError(
                "P2SemanticGuidedDifferenceMapEnhancement expects [P2, P3_semantic]"
            )
        p2_feature, semantic_feature = features

        if self.reconstruction_training_active:
            reconstructed = self.reconstruction_head(p2_feature)
        else:
            with torch.no_grad():
                reconstructed = self.reconstruction_head(p2_feature)
        if reconstructed.shape[-2:] != input_image.shape[-2:]:
            reconstructed = F.interpolate(
                reconstructed,
                size=input_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        target = input_image.float()
        reconstructed_f32 = reconstructed.float()
        self.reconstruction_loss = (
            F.mse_loss(reconstructed_f32, target)
            if self.reconstruction_training_active
            else target.new_zeros(())
        )
        difference = torch.abs(reconstructed_f32 - target).mean(dim=1, keepdim=True)
        if self.threshold_mode == "adaptive_quantile":
            effective_threshold = torch.quantile(
                difference.detach().flatten(1),
                self.adaptive_quantile,
                dim=1,
            ).view(-1, 1, 1, 1)
        else:
            effective_threshold = self.threshold.view(1, 1, 1, 1)
        soft_mask = torch.sigmoid(self.threshold_beta * (difference - effective_threshold))
        hard_mask = (difference >= effective_threshold).to(soft_mask.dtype)
        binary_mask = hard_mask.detach() - soft_mask.detach() + soft_mask
        feature_mask = self.resize_mask(binary_mask, p2_feature.shape[-2:])

        semantic_gate = 2.0 * torch.sigmoid(self.semantic_gate(semantic_feature))
        if semantic_gate.shape[-2:] != p2_feature.shape[-2:]:
            semantic_gate = F.interpolate(
                semantic_gate,
                size=p2_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        final_mask = feature_mask * semantic_gate

        avg_context = F.adaptive_avg_pool2d(p2_feature, output_size=1)
        max_context = F.adaptive_max_pool2d(p2_feature, output_size=1)
        channel_attention = torch.sigmoid(
            self.channel_mlp(avg_context) + self.channel_mlp(max_context)
        )
        alpha = float(getattr(self, "alpha", 0.1))
        enhanced = p2_feature * (1.0 + alpha * channel_attention * final_mask)

        self.last_input_image = target.detach()
        self.last_reconstructed_image = reconstructed_f32.detach()
        self.last_difference_map = difference.detach()
        self.last_binary_map = binary_mask.detach()
        self.last_feature_mask = feature_mask.detach()
        self.last_semantic_gate = semantic_gate.detach()
        self.last_final_mask = final_mask.detach()
        self.last_channel_attention = channel_attention.detach()
        self.last_effective_threshold = effective_threshold.detach()
        output = enhanced if self.enhancement_active else p2_feature
        self.last_feature_relative_change = (
            (output.detach() - p2_feature.detach()).abs().mean()
            / p2_feature.detach().abs().mean().clamp_min(1e-12)
        )
        return output


class DifferenceMapGuidedCrossScaleFusion(P2DifferenceMapEnhancement):
    """Difference Map guided cross-scale fusion from P2 to P3.

    This is the M5-A module. It reconstructs the input image from stride-4 P2,
    builds a sparse Difference Map mask, projects the masked P2 detail to the
    stride-8 P3 grid, and fuses it with the P3 feature through a learnable
    zero-initialized residual gate. The zero gate keeps transferred RT-DETR
    weights stable at initialization, while the direct masked gain provides a
    small immediate DM signal during the screening run.
    """

    def __init__(
        self,
        p2_channels: int,
        target_channels: int,
        recon_weight: float = 1.0,
        threshold_init: float = 0.0156862,
        threshold_beta: float = 50.0,
        reduction: int = 16,
        threshold_mode: str = "adaptive_quantile",
        warmup_epochs: int = 0,
        adaptive_quantile: float = 0.95,
        alpha: float = 0.1,
        mask_resize_mode: str = "nearest",
        detach_mask: bool = True,
        reconstruction_train_epochs: int = -1,
        use_soft_mask: bool = False,
        direct_gain: float = 1.0,
    ):
        super().__init__(
            channels=p2_channels,
            recon_weight=recon_weight,
            threshold_init=threshold_init,
            threshold_beta=threshold_beta,
            reduction=reduction,
            threshold_mode=threshold_mode,
            warmup_epochs=warmup_epochs,
            adaptive_quantile=adaptive_quantile,
            fusion_mode="residual",
            alpha=alpha,
            mask_resize_mode=mask_resize_mode,
            detach_mask=detach_mask,
            reconstruction_train_epochs=reconstruction_train_epochs,
        )
        hidden = max(target_channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(target_channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, target_channels, kernel_size=1, bias=True),
        )
        self.p2_channels = int(p2_channels)
        self.target_channels = int(target_channels)
        self.use_soft_mask = bool(use_soft_mask)
        self.direct_gain = float(direct_gain)
        self.fusion_mode = "dmgcf"
        self.p2_guidance_projection = nn.Sequential(
            nn.Conv2d(
                p2_channels,
                p2_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=p2_channels,
                bias=False,
            ),
            nn.BatchNorm2d(p2_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(p2_channels, target_channels, kernel_size=1, bias=False),
        )
        self.cross_scale_gate = nn.Conv2d(
            target_channels,
            target_channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.zeros_(self.cross_scale_gate.weight)
        self.last_final_mask: torch.Tensor | None = None
        self.last_p2_guidance: torch.Tensor | None = None
        self.last_fused_guidance: torch.Tensor | None = None
        self.last_cross_scale_residual: torch.Tensor | None = None

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        input_image: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) != 2:
            raise TypeError(
                "DifferenceMapGuidedCrossScaleFusion expects [P2, target_P3]"
            )
        p2_feature, target_feature = features

        if self.reconstruction_training_active:
            reconstructed = self.reconstruction_head(p2_feature)
        else:
            with torch.no_grad():
                reconstructed = self.reconstruction_head(p2_feature)
        if reconstructed.shape[-2:] != input_image.shape[-2:]:
            reconstructed = F.interpolate(
                reconstructed,
                size=input_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        target = input_image.float()
        reconstructed_f32 = reconstructed.float()
        self.reconstruction_loss = (
            F.mse_loss(reconstructed_f32, target)
            if self.reconstruction_training_active
            else target.new_zeros(())
        )
        difference = torch.abs(reconstructed_f32 - target).mean(dim=1, keepdim=True)

        if self.threshold_mode == "adaptive_quantile":
            effective_threshold = torch.quantile(
                difference.detach().flatten(1),
                self.adaptive_quantile,
                dim=1,
            ).view(-1, 1, 1, 1)
        else:
            effective_threshold = self.threshold.view(1, 1, 1, 1)
        soft_mask = torch.sigmoid(
            self.threshold_beta * (difference - effective_threshold)
        )
        hard_mask = (difference >= effective_threshold).to(soft_mask.dtype)
        binary_mask = hard_mask.detach() - soft_mask.detach() + soft_mask
        mask_source = soft_mask if self.use_soft_mask else binary_mask
        feature_mask = self.resize_mask(mask_source, target_feature.shape[-2:])
        p2_feature_mask = self.resize_mask(mask_source, p2_feature.shape[-2:])

        avg_context = F.adaptive_avg_pool2d(target_feature, output_size=1)
        max_context = F.adaptive_max_pool2d(target_feature, output_size=1)
        channel_attention = torch.sigmoid(
            self.channel_mlp(avg_context) + self.channel_mlp(max_context)
        )

        p2_guidance = self.p2_guidance_projection(p2_feature * p2_feature_mask)
        if p2_guidance.shape[-2:] != target_feature.shape[-2:]:
            p2_guidance = F.interpolate(
                p2_guidance,
                size=target_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        p3_guidance = target_feature * feature_mask * channel_attention
        fused_guidance = p2_guidance + p3_guidance
        cross_scale_residual = self.cross_scale_gate(fused_guidance)
        alpha = float(getattr(self, "alpha", 0.1))
        direct_gain = float(getattr(self, "direct_gain", 1.0))
        direct_enhanced = target_feature * (
            1.0 + alpha * direct_gain * channel_attention * feature_mask
        )
        enhanced = direct_enhanced + alpha * cross_scale_residual

        self.last_input_image = target.detach()
        self.last_reconstructed_image = reconstructed_f32.detach()
        self.last_difference_map = difference.detach()
        self.last_binary_map = binary_mask.detach()
        self.last_feature_mask = feature_mask.detach()
        self.last_p2_feature_mask = p2_feature_mask.detach()
        self.last_final_mask = feature_mask.detach()
        self.last_p2_guidance = p2_guidance.detach()
        self.last_fused_guidance = fused_guidance.detach()
        self.last_cross_scale_residual = cross_scale_residual.detach()
        self.last_channel_attention = channel_attention.detach()
        self.last_effective_threshold = effective_threshold.detach()
        output = enhanced if self.enhancement_active else target_feature
        self.last_feature_relative_change = (
            (output.detach() - target_feature.detach()).abs().mean()
            / target_feature.detach().abs().mean().clamp_min(1e-12)
        )
        return output


class DifferenceMapGuidedDualMaskFusion(DifferenceMapGuidedCrossScaleFusion):
    """Dual-mask DMGCF that blends conservative P95 and recall-friendly P90 priors.

    M5-A showed that a sparse P95 hard mask reduces false positives but misses
    targets. M5-B showed that a soft P90 mask recovers recall but brings back
    more sea clutter. This module computes both priors and learns a spatial
    gate on the P3 feature grid:

        final_mask = gate * P95_hard + (1 - gate) * P90_soft

    The gate is initialized to 0.5 so the module starts between M5-A and M5-B.
    """

    def __init__(
        self,
        p2_channels: int,
        target_channels: int,
        recon_weight: float = 1.0,
        threshold_init: float = 0.0156862,
        threshold_beta: float = 50.0,
        reduction: int = 16,
        warmup_epochs: int = 0,
        high_quantile: float = 0.95,
        low_quantile: float = 0.90,
        alpha: float = 0.1,
        mask_resize_mode: str = "area",
        detach_mask: bool = True,
        reconstruction_train_epochs: int = -1,
        direct_gain: float = 0.75,
        dm_bbox_loss_weight: float = 0.0,
        dm_bbox_expand: float = 1.2,
    ):
        if not 0.0 < low_quantile < high_quantile < 1.0:
            raise ValueError(
                "Expected 0 < low_quantile < high_quantile < 1, "
                f"got low={low_quantile}, high={high_quantile}"
            )
        super().__init__(
            p2_channels=p2_channels,
            target_channels=target_channels,
            recon_weight=recon_weight,
            threshold_init=threshold_init,
            threshold_beta=threshold_beta,
            reduction=reduction,
            threshold_mode="adaptive_quantile",
            warmup_epochs=warmup_epochs,
            adaptive_quantile=high_quantile,
            alpha=alpha,
            mask_resize_mode=mask_resize_mode,
            detach_mask=detach_mask,
            reconstruction_train_epochs=reconstruction_train_epochs,
            use_soft_mask=False,
            direct_gain=direct_gain,
        )
        hidden = max(target_channels // reduction, 16)
        self.high_quantile = float(high_quantile)
        self.low_quantile = float(low_quantile)
        self.dm_bbox_loss_weight = float(dm_bbox_loss_weight)
        self.dm_bbox_expand = float(dm_bbox_expand)
        self.mask_gate = nn.Sequential(
            nn.Conv2d(target_channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        # sigmoid(0) = 0.5, so the initial mask is a balanced blend.
        nn.init.zeros_(self.mask_gate[-1].weight)
        nn.init.zeros_(self.mask_gate[-1].bias)
        self.fusion_mode = "dual_mask_dmgcf"
        self.last_p95_mask: torch.Tensor | None = None
        self.last_p90_mask: torch.Tensor | None = None
        self.last_mask_gate: torch.Tensor | None = None
        self.last_p90_effective_threshold: torch.Tensor | None = None
        self.dm_supervision_mask: torch.Tensor | None = None

    def _resize_source_mask(
        self,
        mask: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        if getattr(self, "mask_resize_mode", "area") == "area":
            return F.interpolate(mask, size=target_size, mode="area")
        if getattr(self, "mask_resize_mode", "area") == "max":
            if (
                target_size[0] <= mask.shape[-2]
                and target_size[1] <= mask.shape[-1]
            ):
                return F.adaptive_max_pool2d(mask, target_size)
            return F.interpolate(mask, size=target_size, mode="nearest")
        return F.interpolate(mask, size=target_size, mode="nearest")

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        input_image: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) != 2:
            raise TypeError(
                "DifferenceMapGuidedDualMaskFusion expects [P2, target_P3]"
            )
        p2_feature, target_feature = features

        if self.reconstruction_training_active:
            reconstructed = self.reconstruction_head(p2_feature)
        else:
            with torch.no_grad():
                reconstructed = self.reconstruction_head(p2_feature)
        if reconstructed.shape[-2:] != input_image.shape[-2:]:
            reconstructed = F.interpolate(
                reconstructed,
                size=input_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        target = input_image.float()
        reconstructed_f32 = reconstructed.float()
        self.reconstruction_loss = (
            F.mse_loss(reconstructed_f32, target)
            if self.reconstruction_training_active
            else target.new_zeros(())
        )
        difference = torch.abs(reconstructed_f32 - target).mean(dim=1, keepdim=True)

        high_threshold = torch.quantile(
            difference.detach().flatten(1),
            self.high_quantile,
            dim=1,
        ).view(-1, 1, 1, 1)
        low_threshold = torch.quantile(
            difference.detach().flatten(1),
            self.low_quantile,
            dim=1,
        ).view(-1, 1, 1, 1)
        high_soft = torch.sigmoid(self.threshold_beta * (difference - high_threshold))
        high_hard = (difference >= high_threshold).to(high_soft.dtype)
        p95_source = high_hard.detach() - high_soft.detach() + high_soft
        p90_source = torch.sigmoid(self.threshold_beta * (difference - low_threshold))
        p95_detection_source = p95_source.detach() if getattr(self, "detach_mask", False) else p95_source
        p90_detection_source = p90_source.detach() if getattr(self, "detach_mask", False) else p90_source

        p95_feature_mask = self._resize_source_mask(p95_detection_source, target_feature.shape[-2:])
        p90_feature_mask = self._resize_source_mask(p90_detection_source, target_feature.shape[-2:])
        p95_p2_mask = self._resize_source_mask(p95_detection_source, p2_feature.shape[-2:])
        p90_p2_mask = self._resize_source_mask(p90_detection_source, p2_feature.shape[-2:])
        # Keep a non-detached target-aware supervision path. Detection can use
        # detached masks for stability, while the auxiliary bbox-aware loss can
        # still shape the reconstruction/difference branch and the gate.
        p95_supervision_mask = self._resize_source_mask(p95_source, target_feature.shape[-2:])
        p90_supervision_mask = self._resize_source_mask(p90_source, target_feature.shape[-2:])

        mask_gate = torch.sigmoid(self.mask_gate(target_feature))
        if mask_gate.shape[-2:] != target_feature.shape[-2:]:
            mask_gate = F.interpolate(
                mask_gate,
                size=target_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        p2_mask_gate = F.interpolate(
            mask_gate,
            size=p2_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feature_mask = mask_gate * p95_feature_mask + (1.0 - mask_gate) * p90_feature_mask
        p2_feature_mask = p2_mask_gate * p95_p2_mask + (1.0 - p2_mask_gate) * p90_p2_mask
        supervision_mask = mask_gate * p95_supervision_mask + (1.0 - mask_gate) * p90_supervision_mask

        avg_context = F.adaptive_avg_pool2d(target_feature, output_size=1)
        max_context = F.adaptive_max_pool2d(target_feature, output_size=1)
        channel_attention = torch.sigmoid(
            self.channel_mlp(avg_context) + self.channel_mlp(max_context)
        )

        p2_guidance = self.p2_guidance_projection(p2_feature * p2_feature_mask)
        if p2_guidance.shape[-2:] != target_feature.shape[-2:]:
            p2_guidance = F.interpolate(
                p2_guidance,
                size=target_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        p3_guidance = target_feature * feature_mask * channel_attention
        fused_guidance = p2_guidance + p3_guidance
        cross_scale_residual = self.cross_scale_gate(fused_guidance)
        alpha = float(getattr(self, "alpha", 0.1))
        direct_gain = float(getattr(self, "direct_gain", 0.75))
        direct_enhanced = target_feature * (
            1.0 + alpha * direct_gain * channel_attention * feature_mask
        )
        enhanced = direct_enhanced + alpha * cross_scale_residual

        self.last_input_image = target.detach()
        self.last_reconstructed_image = reconstructed_f32.detach()
        self.last_difference_map = difference.detach()
        self.last_binary_map = p95_source.detach()
        self.last_feature_mask = feature_mask.detach()
        self.last_p2_feature_mask = p2_feature_mask.detach()
        self.last_final_mask = feature_mask.detach()
        self.last_p95_mask = p95_feature_mask.detach()
        self.last_p90_mask = p90_feature_mask.detach()
        self.last_mask_gate = mask_gate.detach()
        self.last_p2_guidance = p2_guidance.detach()
        self.last_fused_guidance = fused_guidance.detach()
        self.last_cross_scale_residual = cross_scale_residual.detach()
        self.last_channel_attention = channel_attention.detach()
        self.last_effective_threshold = high_threshold.detach()
        self.last_p90_effective_threshold = low_threshold.detach()
        self.dm_supervision_mask = supervision_mask
        output = enhanced if self.enhancement_active else target_feature
        self.last_feature_relative_change = (
            (output.detach() - target_feature.detach()).abs().mean()
            / target_feature.detach().abs().mean().clamp_min(1e-12)
        )
        return output


class LocalizationAwareDualMaskFusion(DifferenceMapGuidedDualMaskFusion):
    """M7-B localization-aware dual-mask DMGCF.

    M6 improves AP50 by making Difference Map regions target-aware, but the
    error analysis shows many predictions sit in the IoU 0.50-0.75 band: box
    size is often reasonable while the center/boundary is slightly shifted.
    This module keeps the M6 dual-mask design and adds a lightweight P2 detail
    branch gated by the DM/P3 prior, so high-resolution local cues can refine
    the P3 feature before the original three-scale RT-DETR decoder.
    """

    def __init__(
        self,
        p2_channels: int,
        target_channels: int,
        recon_weight: float = 1.0,
        threshold_init: float = 0.0156862,
        threshold_beta: float = 50.0,
        reduction: int = 16,
        warmup_epochs: int = 0,
        high_quantile: float = 0.95,
        low_quantile: float = 0.90,
        alpha: float = 0.1,
        mask_resize_mode: str = "area",
        detach_mask: bool = True,
        reconstruction_train_epochs: int = -1,
        direct_gain: float = 0.75,
        dm_bbox_loss_weight: float = 0.02,
        dm_bbox_expand: float = 1.2,
        localization_gain: float = 0.25,
    ):
        super().__init__(
            p2_channels=p2_channels,
            target_channels=target_channels,
            recon_weight=recon_weight,
            threshold_init=threshold_init,
            threshold_beta=threshold_beta,
            reduction=reduction,
            warmup_epochs=warmup_epochs,
            high_quantile=high_quantile,
            low_quantile=low_quantile,
            alpha=alpha,
            mask_resize_mode=mask_resize_mode,
            detach_mask=detach_mask,
            reconstruction_train_epochs=reconstruction_train_epochs,
            direct_gain=direct_gain,
            dm_bbox_loss_weight=dm_bbox_loss_weight,
            dm_bbox_expand=dm_bbox_expand,
        )
        hidden = max(target_channels // reduction, 16)
        self.localization_gain = float(localization_gain)
        self.p2_local_detail_projection = nn.Sequential(
            nn.Conv2d(
                p2_channels,
                p2_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=p2_channels,
                bias=False,
            ),
            nn.BatchNorm2d(p2_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(p2_channels, target_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(target_channels),
            nn.SiLU(inplace=True),
        )
        self.localization_gate = nn.Sequential(
            nn.Conv2d(target_channels + 1, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.localization_gate[-1].weight)
        nn.init.zeros_(self.localization_gate[-1].bias)
        self.localization_residual_gate = nn.Conv2d(
            target_channels,
            target_channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.zeros_(self.localization_residual_gate.weight)
        self.fusion_mode = "localization_aware_dual_mask_dmgcf"
        self.last_localization_gate: torch.Tensor | None = None
        self.last_p2_local_detail: torch.Tensor | None = None
        self.last_localization_residual: torch.Tensor | None = None

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        input_image: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) != 2:
            raise TypeError(
                "LocalizationAwareDualMaskFusion expects [P2, target_P3]"
            )
        p2_feature, target_feature = features

        if self.reconstruction_training_active:
            reconstructed = self.reconstruction_head(p2_feature)
        else:
            with torch.no_grad():
                reconstructed = self.reconstruction_head(p2_feature)
        if reconstructed.shape[-2:] != input_image.shape[-2:]:
            reconstructed = F.interpolate(
                reconstructed,
                size=input_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        target = input_image.float()
        reconstructed_f32 = reconstructed.float()
        self.reconstruction_loss = (
            F.mse_loss(reconstructed_f32, target)
            if self.reconstruction_training_active
            else target.new_zeros(())
        )
        difference = torch.abs(reconstructed_f32 - target).mean(dim=1, keepdim=True)

        high_threshold = torch.quantile(
            difference.detach().flatten(1),
            self.high_quantile,
            dim=1,
        ).view(-1, 1, 1, 1)
        low_threshold = torch.quantile(
            difference.detach().flatten(1),
            self.low_quantile,
            dim=1,
        ).view(-1, 1, 1, 1)
        high_soft = torch.sigmoid(self.threshold_beta * (difference - high_threshold))
        high_hard = (difference >= high_threshold).to(high_soft.dtype)
        p95_source = high_hard.detach() - high_soft.detach() + high_soft
        p90_source = torch.sigmoid(self.threshold_beta * (difference - low_threshold))
        p95_detection_source = p95_source.detach() if getattr(self, "detach_mask", False) else p95_source
        p90_detection_source = p90_source.detach() if getattr(self, "detach_mask", False) else p90_source

        p95_feature_mask = self._resize_source_mask(p95_detection_source, target_feature.shape[-2:])
        p90_feature_mask = self._resize_source_mask(p90_detection_source, target_feature.shape[-2:])
        p95_p2_mask = self._resize_source_mask(p95_detection_source, p2_feature.shape[-2:])
        p90_p2_mask = self._resize_source_mask(p90_detection_source, p2_feature.shape[-2:])
        p95_supervision_mask = self._resize_source_mask(p95_source, target_feature.shape[-2:])
        p90_supervision_mask = self._resize_source_mask(p90_source, target_feature.shape[-2:])

        mask_gate = torch.sigmoid(self.mask_gate(target_feature))
        if mask_gate.shape[-2:] != target_feature.shape[-2:]:
            mask_gate = F.interpolate(
                mask_gate,
                size=target_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        p2_mask_gate = F.interpolate(
            mask_gate,
            size=p2_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feature_mask = mask_gate * p95_feature_mask + (1.0 - mask_gate) * p90_feature_mask
        p2_feature_mask = p2_mask_gate * p95_p2_mask + (1.0 - p2_mask_gate) * p90_p2_mask
        supervision_mask = mask_gate * p95_supervision_mask + (1.0 - mask_gate) * p90_supervision_mask

        avg_context = F.adaptive_avg_pool2d(target_feature, output_size=1)
        max_context = F.adaptive_max_pool2d(target_feature, output_size=1)
        channel_attention = torch.sigmoid(
            self.channel_mlp(avg_context) + self.channel_mlp(max_context)
        )

        p2_guidance = self.p2_guidance_projection(p2_feature * p2_feature_mask)
        if p2_guidance.shape[-2:] != target_feature.shape[-2:]:
            p2_guidance = F.interpolate(
                p2_guidance,
                size=target_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        p3_guidance = target_feature * feature_mask * channel_attention
        fused_guidance = p2_guidance + p3_guidance
        cross_scale_residual = self.cross_scale_gate(fused_guidance)

        p2_local_detail = self.p2_local_detail_projection(p2_feature * p2_feature_mask)
        if p2_local_detail.shape[-2:] != target_feature.shape[-2:]:
            p2_local_detail = F.interpolate(
                p2_local_detail,
                size=target_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        localization_gate = torch.sigmoid(
            self.localization_gate(torch.cat([target_feature, feature_mask], dim=1))
        )
        local_detail = p2_local_detail * localization_gate * channel_attention
        localization_residual = self.localization_residual_gate(local_detail)

        alpha = float(getattr(self, "alpha", 0.1))
        direct_gain = float(getattr(self, "direct_gain", 0.75))
        localization_gain = float(getattr(self, "localization_gain", 0.25))
        direct_enhanced = target_feature * (
            1.0 + alpha * direct_gain * channel_attention * feature_mask
        )
        enhanced = (
            direct_enhanced
            + alpha * cross_scale_residual
            + alpha * localization_gain * (local_detail + localization_residual)
        )

        self.last_input_image = target.detach()
        self.last_reconstructed_image = reconstructed_f32.detach()
        self.last_difference_map = difference.detach()
        self.last_binary_map = p95_source.detach()
        self.last_feature_mask = feature_mask.detach()
        self.last_p2_feature_mask = p2_feature_mask.detach()
        self.last_final_mask = feature_mask.detach()
        self.last_p95_mask = p95_feature_mask.detach()
        self.last_p90_mask = p90_feature_mask.detach()
        self.last_mask_gate = mask_gate.detach()
        self.last_p2_guidance = p2_guidance.detach()
        self.last_fused_guidance = fused_guidance.detach()
        self.last_cross_scale_residual = cross_scale_residual.detach()
        self.last_channel_attention = channel_attention.detach()
        self.last_effective_threshold = high_threshold.detach()
        self.last_p90_effective_threshold = low_threshold.detach()
        self.dm_supervision_mask = supervision_mask
        self.last_localization_gate = localization_gate.detach()
        self.last_p2_local_detail = p2_local_detail.detach()
        self.last_localization_residual = localization_residual.detach()
        output = enhanced if self.enhancement_active else target_feature
        self.last_feature_relative_change = (
            (output.detach() - target_feature.detach()).abs().mean()
            / target_feature.detach().abs().mean().clamp_min(1e-12)
        )
        return output


class P2GuidedDifferenceMapEnhancement(DifferenceMapGuidedEnhancement):
    """Reconstruct from stride-4 P2 and enhance the OWRT stride-8 P3 feature.

    This keeps the RT-DETR decoder interface unchanged: the module receives
    ``[P2, P3]`` and returns an enhanced P3 tensor. P2 is used only to create
    the reconstruction and Difference Map prior.
    """

    def __init__(
        self,
        p2_channels: int,
        target_channels: int,
        recon_weight: float = 0.1,
        threshold_init: float = 0.0156862,
        threshold_beta: float = 50.0,
        reduction: int = 16,
        threshold_mode: str = "adaptive_quantile",
        warmup_epochs: int = 2,
        adaptive_quantile: float = 0.95,
        fusion_mode: str = "residual",
        alpha: float = 0.1,
        mask_resize_mode: str = "nearest",
        detach_mask: bool = False,
        detach_reconstruction_input: bool = False,
        use_soft_mask: bool = False,
        reconstruction_train_epochs: int = -1,
        use_semantic_gate: bool = False,
    ):
        super().__init__(
            channels=target_channels,
            recon_weight=recon_weight,
            threshold_init=threshold_init,
            threshold_beta=threshold_beta,
            reduction=reduction,
            threshold_mode=threshold_mode,
            warmup_epochs=warmup_epochs,
            adaptive_quantile=adaptive_quantile,
            fusion_mode=fusion_mode,
            alpha=alpha,
            mask_resize_mode=mask_resize_mode,
            detach_mask=detach_mask,
            reconstruction_train_epochs=reconstruction_train_epochs,
        )
        c1 = max(p2_channels // 2, 64)
        c2 = max(c1 // 2, 32)
        self.reconstruction_head = nn.Sequential(
            ReconstructionUpBlock(p2_channels, c1),
            ReconstructionUpBlock(c1, c2),
            nn.Conv2d(c2, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.p2_channels = int(p2_channels)
        self.target_channels = int(target_channels)
        self.detach_reconstruction_input = bool(detach_reconstruction_input)
        self.use_soft_mask = bool(use_soft_mask)
        self.use_semantic_gate = bool(use_semantic_gate)
        self.p2_guidance_projection = nn.Sequential(
            nn.Conv2d(
                p2_channels,
                p2_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=p2_channels,
                bias=False,
            ),
            nn.BatchNorm2d(p2_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(p2_channels, target_channels, kernel_size=1, bias=False),
        )
        # A zero convolution is a learnable residual gate. It starts as the
        # exact alpha=0 control, receives gradients immediately, and can learn
        # positive or negative corrections rather than only amplifying P3.
        self.zero_gate = nn.Conv2d(
            target_channels,
            target_channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.zeros_(self.zero_gate.weight)
        semantic_hidden = max(target_channels // 4, 32)
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(target_channels, semantic_hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(semantic_hidden, 1, kernel_size=1, bias=True),
        )
        # 2 * sigmoid(0) = 1, so enabling the semantic gate starts as an
        # exact identity modulation and does not disturb transferred weights.
        nn.init.zeros_(self.semantic_gate[-1].weight)
        nn.init.zeros_(self.semantic_gate[-1].bias)
        self.last_semantic_gate: torch.Tensor | None = None

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        input_image: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) not in {2, 3}:
            raise TypeError(
                "P2GuidedDifferenceMapEnhancement expects [P2, target_P3] "
                "or [P2, cross_scale_P3, multiscale_P3]"
            )
        p2_feature = features[0]
        target_feature = (
            features[1] if len(features) == 2 else features[1] + features[2]
        )
        reconstruction_input = (
            p2_feature.detach()
            if getattr(self, "detach_reconstruction_input", False)
            else p2_feature
        )
        if self.reconstruction_training_active:
            reconstructed = self.reconstruction_head(reconstruction_input)
        else:
            with torch.no_grad():
                reconstructed = self.reconstruction_head(reconstruction_input)
        if reconstructed.shape[-2:] != input_image.shape[-2:]:
            reconstructed = F.interpolate(
                reconstructed,
                size=input_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        target = input_image.float()
        reconstructed_f32 = reconstructed.float()
        self.reconstruction_loss = (
            F.mse_loss(reconstructed_f32, target)
            if self.reconstruction_training_active
            else target.new_zeros(())
        )
        difference = torch.abs(reconstructed_f32 - target).mean(dim=1, keepdim=True)

        if self.threshold_mode == "adaptive_quantile":
            effective_threshold = torch.quantile(
                difference.detach().flatten(1),
                self.adaptive_quantile,
                dim=1,
            ).view(-1, 1, 1, 1)
        else:
            effective_threshold = self.threshold.view(1, 1, 1, 1)
        soft_mask = torch.sigmoid(
            self.threshold_beta * (difference - effective_threshold)
        )
        hard_mask = (difference >= effective_threshold).to(soft_mask.dtype)
        binary_mask = hard_mask.detach() - soft_mask.detach() + soft_mask
        mask_source = (
            soft_mask
            if getattr(self, "use_soft_mask", False)
            else binary_mask
        )
        feature_mask = self.resize_mask(mask_source, target_feature.shape[-2:])
        p2_feature_mask = self.resize_mask(mask_source, p2_feature.shape[-2:])

        avg_context = F.adaptive_avg_pool2d(target_feature, output_size=1)
        max_context = F.adaptive_max_pool2d(target_feature, output_size=1)
        channel_attention = torch.sigmoid(
            self.channel_mlp(avg_context) + self.channel_mlp(max_context)
        )

        fusion_mode = getattr(self, "fusion_mode", "residual")
        alpha = float(getattr(self, "alpha", 0.1))
        gated_residual = None
        if fusion_mode == "gated_residual":
            semantic_gate = (
                2.0 * torch.sigmoid(self.semantic_gate(target_feature))
                if getattr(self, "use_semantic_gate", False)
                else torch.ones_like(feature_mask)
            )
            p2_guided = self.p2_guidance_projection(
                p2_feature * p2_feature_mask
            )
            if p2_guided.shape[-2:] != target_feature.shape[-2:]:
                p2_guided = F.interpolate(
                    p2_guided,
                    size=target_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            p2_guided = p2_guided * semantic_gate
            p3_guided = target_feature * feature_mask * semantic_gate
            gated_residual = self.zero_gate(p3_guided + p2_guided)
            enhanced = target_feature + alpha * gated_residual
        elif fusion_mode == "residual":
            enhanced = target_feature * (
                1.0 + alpha * channel_attention * feature_mask
            )
        else:
            enhanced = target_feature * channel_attention
            enhanced = enhanced * (1.0 + feature_mask)

        self.last_input_image = target.detach()
        self.last_reconstructed_image = reconstructed_f32.detach()
        self.last_difference_map = difference.detach()
        self.last_binary_map = binary_mask.detach()
        self.last_feature_mask = feature_mask.detach()
        self.last_p2_feature_mask = p2_feature_mask.detach()
        self.last_gated_residual = (
            None if gated_residual is None else gated_residual.detach()
        )
        self.last_semantic_gate = (
            None
            if fusion_mode != "gated_residual"
            else semantic_gate.detach()
        )
        self.last_channel_attention = channel_attention.detach()
        self.last_effective_threshold = effective_threshold.detach()
        output = enhanced if self.enhancement_active else target_feature
        self.last_feature_relative_change = (
            (output.detach() - target_feature.detach()).abs().mean()
            / target_feature.detach().abs().mean().clamp_min(1e-12)
        )
        return output



class LiteDepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable 3x3 conv used by LiteRH320."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                      groups=in_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PaperOWRTDMDGFELiteRH320(PaperOWRTDMDGFE):
    """Lightweight Paper-adaptive DGFE with 320x320 reconstruction.

    Difference Map logic is kept:
        D = mean(abs(Recon - Input_resized))
        Db = adaptive quantile threshold
        P2' = P2 * channel_att * (Db_p2 + 1)

    Compared with PaperOWRTDMDGFE:
        - only reconstructs to 320x320 for 640 input;
        - replaces heavy ConvTranspose/Conv RH with Upsample + depthwise separable conv;
        - keeps q=0.97 adaptive filtration and Paper-DGFE formula.
    """

    def __init__(
        self,
        channels: int,
        recon_weight: float = 0.1,
        threshold_init: float = 0.0156862,
        threshold_beta: float = 50.0,
        reduction: int = 16,
        paper_strict: bool = True,
        reconstruction_train_epochs: int = -1,
        threshold_mode: str = "adaptive_quantile",
        adaptive_quantile: float = 0.97,
    ):
        super().__init__(
            channels=channels,
            recon_weight=recon_weight,
            threshold_init=threshold_init,
            threshold_beta=threshold_beta,
            reduction=reduction,
            paper_strict=paper_strict,
            reconstruction_train_epochs=reconstruction_train_epochs,
            threshold_mode=threshold_mode,
            adaptive_quantile=adaptive_quantile,
        )
        c1 = max(channels // 2, 64)
        c2 = max(c1 // 2, 32)

        self.reconstruction_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            LiteDepthwiseSeparableConv2d(channels, c1),
            LiteDepthwiseSeparableConv2d(c1, c2),
            nn.Conv2d(c2, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, p2_feature: torch.Tensor, input_image: torch.Tensor) -> torch.Tensor:
        if self.reconstruction_training_active:
            reconstructed = self.reconstruction_head(p2_feature)
        else:
            with torch.no_grad():
                reconstructed = self.reconstruction_head(p2_feature)

        raw_img_01 = input_image.float()
        reconstructed_f32 = reconstructed.float()

        if raw_img_01.shape[-2:] != reconstructed_f32.shape[-2:]:
            raw_for_diff = F.interpolate(
                raw_img_01,
                size=reconstructed_f32.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            raw_for_diff = raw_img_01

        self.reconstruction_loss = (
            F.mse_loss(reconstructed_f32, raw_for_diff)
            if self.reconstruction_training_active
            else raw_img_01.new_zeros(())
        )

        difference = torch.abs(reconstructed_f32 - raw_for_diff).mean(dim=1, keepdim=True)

        if getattr(self, "threshold_mode", "learnable") == "adaptive_quantile":
            threshold = torch.quantile(
                difference.detach().flatten(1),
                float(getattr(self, "adaptive_quantile", 0.97)),
                dim=1,
            ).view(-1, 1, 1, 1)
        else:
            threshold = self.threshold.view(1, 1, 1, 1)

        threshold = threshold.to(device=difference.device, dtype=difference.dtype)

        if self.paper_strict:
            db = (torch.sign(difference - threshold) + 1.0) * 0.5
        else:
            soft = torch.sigmoid(float(getattr(self, "threshold_beta", 50.0)) * (difference - threshold))
            hard = (difference >= threshold).to(soft.dtype)
            db = hard.detach() - soft.detach() + soft

        db_p2 = F.interpolate(db, size=p2_feature.shape[-2:], mode="nearest")
        filtration = db_p2 + 1.0

        kernel_size = p2_feature.shape[-2:]
        avg_context = F.avg_pool2d(p2_feature, kernel_size=kernel_size)
        max_context = F.max_pool2d(p2_feature, kernel_size=kernel_size)
        channel_reweight = torch.sigmoid(self.channel_mlp(avg_context) + self.channel_mlp(max_context))

        attention_matrix = channel_reweight * filtration
        p2_prime = p2_feature * attention_matrix
        output = p2_prime if self.enhancement_active else p2_feature

        self.last_input_image = raw_for_diff.detach()
        self.last_reconstructed_image = reconstructed_f32.detach()
        self.last_difference_map = difference.detach()
        self.last_binary_map = db.detach()
        self.last_feature_mask = filtration.detach()
        self.last_filtration_map = filtration.detach()
        self.last_channel_reweight = channel_reweight.detach()
        self.last_attention_matrix = attention_matrix.detach()
        self.last_p2_input = p2_feature.detach()
        self.last_p2_enhanced = output.detach()
        self.last_p2_residual = output.detach() - p2_feature.detach()
        self.last_effective_threshold = threshold.detach()
        self.last_feature_relative_change = (
            (output.detach() - p2_feature.detach()).abs().mean()
            / p2_feature.detach().abs().mean().clamp_min(1e-12)
        )
        return output
