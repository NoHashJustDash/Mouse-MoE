# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact parameter count of ``model.MouseMoe`` from its config, without PyTorch.

Kept in lock-step with ``model.py``; ``tests/test_model.py`` asserts equality
with the instantiated model's ``parameter_count``.
"""

from __future__ import annotations

from mouse_moe.config import ModelConfig

TOKENS = 64
INPUT_PLANES = 112
PROMOTION_CHANNELS = 4


def _linear(inputs: int, outputs: int, bias: bool = True) -> int:
    return inputs * outputs + (outputs if bias else 0)


def _ffn(d: int, hidden: int) -> int:
    return _linear(d, hidden) + _linear(hidden, d)


def parameter_breakdown(cfg: ModelConfig) -> dict[str, int]:
    d, h = cfg.d_model, cfg.heads
    g = cfg.gab
    parts = {
        "embedding": _linear(INPUT_PLANES, d) + TOKENS * d + d,
        "gab_shared": g.per_head * TOKENS * TOKENS,
        "attention": 0,
        "gab": 0,
        "norms": 0,
        "dense_ffn": 0,
        "moe": 0,
    }
    per_gab = (
        _linear(d, g.compress, bias=False)
        + _linear(TOKENS * g.compress, g.hidden)
        + g.hidden
        + _linear(g.hidden, h * g.per_head)
        + g.per_head
    )
    m = cfg.moe
    experts = m.experts * _ffn(d, m.expert_hidden)
    shared = _ffn(d, m.shared_hidden) if m.shared_hidden else 0
    for index in range(cfg.layers):
        parts["attention"] += 3 * d * d + _linear(d, d)
        parts["gab"] += per_gab
        parts["norms"] += 2 * d
        moe_here = cfg.ffn == "moe" and index % m.every == m.every - 1
        if not moe_here:
            parts["dense_ffn"] += _ffn(d, cfg.dense_hidden)
        elif m.routing == "soft":
            parts["moe"] += d * m.experts * m.slots_per_expert + 1 + experts + shared
        else:
            parts["moe"] += d * m.experts + experts + shared
    parts["policy_head"] = 2 * _linear(d, d) + _linear(d, PROMOTION_CHANNELS)
    parts["value_head"] = (
        _linear(d, cfg.value_channels)
        + _linear(TOKENS * cfg.value_channels, cfg.value_hidden)
        + _linear(cfg.value_hidden, 3)
    )
    parts["moves_left_head"] = (
        _linear(d, cfg.mlh_channels)
        + _linear(TOKENS * cfg.mlh_channels, cfg.mlh_hidden)
        + _linear(cfg.mlh_hidden, 1)
    )
    return parts


def parameter_total(cfg: ModelConfig) -> int:
    return sum(parameter_breakdown(cfg).values())


def macs_per_position(cfg: ModelConfig) -> int:
    """Approximate multiply-accumulates per evaluated position (body + attention maps)."""
    from mouse_moe.config import MoeConfig  # noqa: F401  (documents the dependency)

    d = cfg.d_model
    m = cfg.moe
    total = 0
    for index in range(cfg.layers):
        total += TOKENS * 4 * d * d + 2 * TOKENS * TOKENS * d
        moe_here = cfg.ffn == "moe" and index % m.every == m.every - 1
        if not moe_here:
            total += TOKENS * 2 * d * cfg.dense_hidden
        elif m.routing == "soft":
            slots = m.experts * m.slots_per_expert
            total += 2 * TOKENS * slots * d + slots * 2 * d * m.expert_hidden
            total += TOKENS * 2 * d * m.shared_hidden
        else:
            total += TOKENS * (2 * d * m.shared_hidden + m.top_k * 2 * d * m.expert_hidden)
            total += TOKENS * d * m.experts
    return total
