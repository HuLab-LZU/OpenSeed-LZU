"""Loss functions and class-weight helpers for long-tail classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss (multiclass).

    Arguments
    ---------
    gamma : Modulation exponent (default 2.0).
    alpha : Optional focal weight. Can be a scalar float or a per-class 1-D
            tensor. ``None`` uses the uniform version.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor) and self.alpha.ndim == 1:
                alpha_t = self.alpha[target]
            else:
                alpha_t = self.alpha
            loss = alpha_t * loss

        return loss.mean()


def build_class_weights(
    class_counts: torch.Tensor,
    mode: str,
    beta: float = 0.9999,
) -> torch.Tensor | None:
    """Return per-class loss weights for a given long-tail strategy.

    modes
    -----
    none           -> None
    inverse_freq   -> total / (num_classes * count)
    class_balanced -> (1 - beta) / (1 - beta^count)  (Cui et al.)
    """
    if mode == "none":
        return None
    if class_counts is None:
        return None

    counts = class_counts.clamp(min=1).float()
    num_classes = class_counts.numel()
    total = class_counts.sum().float()

    if mode == "inverse_freq":
        weights = total / (num_classes * counts)
    elif mode == "class_balanced":
        weights = (1.0 - beta) / (1.0 - beta**counts)
    else:
        raise ValueError(f"Unknown class-weight mode: {mode}")

    # Normalize so weights have mean 1; the absolute scale does not affect
    # the loss gradient direction, only the effective learning-rate scale.
    weights = weights / weights.mean()
    return weights
