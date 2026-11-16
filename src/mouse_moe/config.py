# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed, validated run configuration loaded from TOML (stdlib ``tomllib``)."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

FfnKind = Literal["dense", "moe"]
Routing = Literal["token", "position", "soft"]


class ConfigError(ValueError):
    """The configuration is incomplete, inconsistent or names unknown keys."""


@dataclass(frozen=True)
class MoeConfig:
    experts: int = 16
    top_k: int = 2
    expert_hidden: int = 192
    shared_hidden: int = 384
    routing: Routing = "token"
    every: int = 1  # 1: every layer is MoE; 2: alternate layers (others dense)
    slots_per_expert: int = 4  # soft routing only
    bias_update_rate: float = 1e-3  # aux-loss-free balancing (token/position routing)
    aux_loss_weight: float = 1e-4
    z_loss_weight: float = 1e-3


@dataclass(frozen=True)
class GabConfig:
    compress: int = 32  # d1: per-token compression width
    hidden: int = 128  # d2: board summary width
    per_head: int = 128  # d3: per-head generator width


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 384
    layers: int = 12
    heads: int = 12
    ffn: FfnKind = "moe"
    dense_hidden: int = 768
    moe: MoeConfig = field(default_factory=MoeConfig)
    gab: GabConfig = field(default_factory=GabConfig)
    value_channels: int = 32
    value_hidden: int = 128
    mlh_channels: int = 8
    mlh_hidden: int = 128
    parameter_budget: int = 50_000_000


@dataclass(frozen=True)
class DataConfig:
    directory: str = "data/test80"
    train_shards: tuple[str, ...] = ()
    validation_shards: tuple[str, ...] = ()
    test_shards: tuple[str, ...] = ()  # untouched until final evaluation
    validation_positions_per_game: int = 16
    workers: int = 6
    readers_per_worker: int = 4
    shuffle_buffer: int = 250_000  # total across workers
    sample_rate: float = 0.5
    verify_hashes: bool = False
    # Teacher label directory (relative to the repository root), mixed into the training targets:
    # target = teacher_weight * teacher + (1 - teacher_weight) * game (see teacher.py).
    teacher_labels: str = ""
    teacher_weight: float = 1.0
    teacher_temperature: float = 1.0
    # Mirror castling-free training positions file-wise at random (augment.py).
    mirror: bool = False
    # Training keeps plies before opening_plies at rate opening_keep (repeated openings).
    opening_plies: int = 0
    opening_keep: float = 1.0
    # More training shards, from a second verified download (relative to the repository root),
    # trained on their games' own targets even when teacher_labels is set, unless
    # game_teacher_labels names their own label directory: then they mix in those labels
    # with teacher_weight, like the shards in `directory`.
    game_directory: str = ""
    game_shards: tuple[str, ...] = ()
    game_teacher_labels: str = ""


@dataclass(frozen=True)
class LossConfig:
    policy: float = 1.0
    wdl: float = 1.0
    q_wdl: float = 0.5
    moves_left: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 1024  # positions per optimizer step
    grad_accum_steps: int = 1  # micro-batches of batch_size / grad_accum_steps per step
    steps: int = 400_000
    warmup_steps: int = 2_000
    lr: float = 5e-4
    lr_boundaries: tuple[int, ...] = (320_000, 370_000)
    lr_factors: tuple[float, ...] = (0.316, 0.1)
    # Re-warm for a fork: from step rewarm_start the factor ramps linearly from rewarm_from
    # to the scheduled one over rewarm_steps steps (0: no re-warm).
    rewarm_start: int = 0
    rewarm_steps: int = 0
    rewarm_from: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-7
    weight_decay: float = 1e-4
    grad_clip: float = 10.0
    precision: Literal["bf16", "fp32"] = "bf16"
    swa_start: int = 100_000
    swa_every: int = 100
    validate_every: int = 5_000
    validation_batches: int = 50
    checkpoint_every: int = 5_000
    log_every: int = 100
    balance_calibration_batches: int = 300  # re-tune SWA routing biases at the end
    seed: int = 1
    compile: bool = False


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: ModelConfig
    data: DataConfig
    loss: LossConfig
    train: TrainConfig


def _build(cls: type[Any], values: dict[str, Any], where: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ConfigError(f"unknown key(s) in [{where}]: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _build(type(default), value, f"{where}.{name}")
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def validate(config: RunConfig) -> None:
    m, d, t = config.model, config.data, config.train
    if m.d_model % m.heads:
        raise ConfigError("d_model must be divisible by heads")
    if m.layers < 1 or m.heads < 1:
        raise ConfigError("layers and heads must be positive")
    if m.ffn == "moe":
        if not 1 <= m.moe.top_k <= m.moe.experts:
            raise ConfigError("moe.top_k must be in 1..experts")
        if m.moe.every < 1:
            raise ConfigError("moe.every must be >= 1")
        if m.moe.routing == "soft" and m.moe.slots_per_expert < 1:
            raise ConfigError("soft routing needs slots_per_expert >= 1")
    if not d.train_shards or not d.validation_shards:
        raise ConfigError("data.train_shards and data.validation_shards must be non-empty")
    sets = (set(d.train_shards), set(d.validation_shards), set(d.test_shards))
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ConfigError("a shard cannot be in both training and validation/test sets")
    if bool(d.game_directory) != bool(d.game_shards):
        raise ConfigError("data.game_directory and data.game_shards are set together")
    if set(d.game_shards) & (sets[0] | sets[1] | sets[2]):
        raise ConfigError("a game shard cannot share a name with another data shard")
    if d.game_teacher_labels and not d.game_shards:
        raise ConfigError("data.game_teacher_labels needs data.game_shards")
    if len(d.validation_shards) != 1:
        raise ConfigError("exactly one validation shard is supported")
    if d.validation_positions_per_game < 1:
        raise ConfigError("validation_positions_per_game must be >= 1")
    if len(t.lr_boundaries) != len(t.lr_factors):
        raise ConfigError("lr_boundaries and lr_factors must have equal length")
    if list(t.lr_boundaries) != sorted(t.lr_boundaries):
        raise ConfigError("lr_boundaries must be increasing")
    if t.rewarm_steps < 0 or (t.rewarm_steps and t.rewarm_from <= 0.0):
        raise ConfigError("rewarm_steps must be >= 0 and rewarm_from positive")
    if t.rewarm_steps and t.rewarm_start < t.warmup_steps:
        raise ConfigError("a re-warm must start after the warm-up")
    if t.grad_accum_steps < 1 or t.batch_size % (8 * t.grad_accum_steps):
        raise ConfigError("batch_size must be a multiple of 8 x grad_accum_steps")
    if d.workers < 1 or d.shuffle_buffer // d.workers < t.batch_size:
        raise ConfigError("each worker's share of shuffle_buffer must hold one batch")
    if not 0.0 < d.sample_rate <= 1.0 or abs(1 / d.sample_rate - round(1 / d.sample_rate)) > 1e-9:
        raise ConfigError("data.sample_rate must be 1/n for a whole number n (0.5, 0.25, ...)")
    if t.balance_calibration_batches < 0:
        raise ConfigError("balance_calibration_batches must be >= 0")


def load_config(path: Path) -> RunConfig:
    with path.open("rb") as stream:
        return load_config_dict(tomllib.load(stream))


def load_config_dict(raw: dict[str, Any]) -> RunConfig:
    """Build and validate a config from a TOML table or a saved ``to_dict`` result."""
    if "name" not in raw:
        raise ConfigError("config needs a top-level name")
    sections = {"model": ModelConfig, "data": DataConfig, "loss": LossConfig, "train": TrainConfig}
    unknown = set(raw) - set(sections) - {"name"}
    if unknown:
        raise ConfigError(f"unknown section(s): {sorted(unknown)}")
    built = {key: _build(cls, raw.get(key, {}), key) for key, cls in sections.items()}
    config = RunConfig(name=str(raw["name"]), **built)
    validate(config)
    return config


def to_dict(config: RunConfig) -> dict[str, Any]:
    return asdict(config)
