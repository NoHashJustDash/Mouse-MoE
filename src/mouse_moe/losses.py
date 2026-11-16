# SPDX-License-Identifier: GPL-3.0-or-later
"""Training losses and metrics for policy, WDL value, search-Q value and moves-left."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from mouse_moe.config import LossConfig

MLH_SCALE = 20.0  # plies; Huber on scaled values keeps the term O(1)


@dataclass
class LossTerms:
    total: Tensor
    policy: Tensor
    wdl: Tensor
    q_wdl: Tensor
    moves_left: Tensor
    moe_aux: Tensor

    def as_floats(self) -> dict[str, float]:
        return {name: float(value.detach()) for name, value in vars(self).items()}


def masked_log_policy(logits: Tensor, target: Tensor) -> Tensor:
    """log-softmax over legal moves only (Lc0 marks illegal moves with a negative target)."""
    legal = target >= 0
    masked = logits.float().masked_fill(~legal, float("-inf"))
    return torch.where(legal, masked.log_softmax(dim=-1), torch.zeros_like(masked))


def policy_cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    return -(target.clamp(min=0) * masked_log_policy(logits, target)).sum(dim=-1).mean()


def soft_cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    return -(target * logits.float().log_softmax(dim=-1)).sum(dim=-1).mean()


def compute_losses(
    outputs: tuple[Tensor, Tensor, Tensor],
    batch: dict[str, Tensor],
    weights: LossConfig,
    moe_aux: Tensor,
) -> LossTerms:
    policy_logits, wdl_logits, mlh = outputs
    policy = policy_cross_entropy(policy_logits, batch["policy"])
    wdl = soft_cross_entropy(wdl_logits, batch["wdl"])
    q_wdl = soft_cross_entropy(wdl_logits, batch["q_wdl"])
    moves_left = F.huber_loss(
        mlh.float().squeeze(1) / MLH_SCALE, batch["plies_left"] / MLH_SCALE, delta=0.5
    )
    total = (
        weights.policy * policy
        + weights.wdl * wdl
        + weights.q_wdl * q_wdl
        + weights.moves_left * moves_left
        + moe_aux
    )
    return LossTerms(total, policy, wdl, q_wdl, moves_left, moe_aux)


@torch.no_grad()
def metrics(outputs: tuple[Tensor, Tensor, Tensor], batch: dict[str, Tensor]) -> dict[str, float]:
    policy_logits, wdl_logits, mlh = outputs
    target = batch["policy"]
    legal = target >= 0
    masked = policy_logits.float().masked_fill(~legal, float("-inf"))
    top1 = (masked.argmax(dim=-1) == target.argmax(dim=-1)).float().mean()
    result = batch["wdl"].argmax(dim=-1)
    value_accuracy = (wdl_logits.argmax(dim=-1) == result).float().mean()
    mlh_error = (mlh.float().squeeze(1) - batch["plies_left"]).abs().mean()
    return {
        "policy_top1": float(top1),
        "value_accuracy": float(value_accuracy),
        "mlh_mae_plies": float(mlh_error),
    }
