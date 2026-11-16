# SPDX-License-Identifier: GPL-3.0-or-later
"""Train a Mouse-MoE network on verified Lc0 V6 shards.

    python -I -m mouse_moe.train --config configs/m1_46m.toml
    python -I -m mouse_moe.train --config configs/m1_46m.toml --resume   # continue latest

Outputs go to ``runs/<name>/``: ``config.json``, ``metrics.jsonl`` (one JSON object
per log/validation event), ``checkpoints/step_<n>.pt`` and ``checkpoints/latest.pt``.
The run refuses to start without CUDA unless ``--device cpu`` is given explicitly.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from mouse_moe.config import RunConfig, load_config, to_dict
from mouse_moe.dataset import StreamSpec, build_validation_set, stream_batches
from mouse_moe.losses import compute_losses, metrics
from mouse_moe.model import MouseMoe, RoutedMoe, expand_planes, parameter_count
from mouse_moe.resources import MemoryGuard, MemoryReserveError, no_idle_sleep
from mouse_moe.shards import verified_shards
from mouse_moe.teacher import require_labels
from mouse_moe.v6 import Batch, take

ROOT = Path(__file__).resolve().parents[2]


def _batch_to_tensors(batch: Batch) -> dict[str, torch.Tensor]:
    return {
        "packed": torch.from_numpy(batch.packed),
        "aux": torch.from_numpy(batch.aux),
        "policy": torch.from_numpy(batch.policy),
        "wdl": torch.from_numpy(batch.wdl),
        "q_wdl": torch.from_numpy(batch.q_wdl),
        "plies_left": torch.from_numpy(batch.plies_left),
    }


class ShardDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Each DataLoader worker streams its own disjoint stride of games from every shard."""

    def __init__(self, shards: list[Path], config: RunConfig, seed: int, train: bool) -> None:
        self.shards, self.config, self.seed, self.train = shards, config, seed, train

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        d, t = self.config.data, self.config.train
        spec = StreamSpec(
            shards=tuple(self.shards),
            batch_size=micro_batch(self.config),
            shuffle_buffer=max(t.batch_size, d.shuffle_buffer // workers)
            if self.train
            else t.batch_size * 4,
            sample_rate=d.sample_rate if self.train else 1.0,
            readers=d.readers_per_worker if self.train else 1,
            cycle=self.train,
            seed=self.seed * 1009 + worker,
            worker=worker,
            workers=workers,
            # Validation always scores against the games' own targets.
            teacher_labels=ROOT / d.teacher_labels
            if d.teacher_labels and self.train
            else None,
            teacher_weight=d.teacher_weight,
            teacher_temperature=d.teacher_temperature,
            mirror=d.mirror and self.train,
            opening_plies=d.opening_plies if self.train else 0,
            opening_keep=d.opening_keep if self.train else 1.0,
            game_only=frozenset(d.game_shards),
            game_labels=ROOT / d.game_teacher_labels
            if d.game_teacher_labels and self.train
            else None,
        )
        for batch in stream_batches(spec):
            yield _batch_to_tensors(batch)


def game_shards(config: RunConfig) -> list[Path]:
    """The verified ``data.game_shards`` (trained on their games' own targets)."""
    d = config.data
    if not d.game_shards:
        return []
    directory = (ROOT / d.game_directory).resolve()
    return verified_shards(directory, list(d.game_shards), d.verify_hashes)


def micro_batch(config: RunConfig) -> int:
    return config.train.batch_size // config.train.grad_accum_steps


def validation_batches(config: RunConfig, shard: Path) -> list[dict[str, torch.Tensor]]:
    """Fixed validation sample spread over the whole held-out shard, cached on disk.

    The sample holds ``validation_batches x batch_size`` positions whatever the
    gradient accumulation; it is evaluated in micro-batches.
    """
    t, d = config.train, config.data
    positions = t.validation_batches * t.batch_size
    cache = (
        shard.parent
        / "cache"
        / (f"{shard.stem}-val{positions}-pg{d.validation_positions_per_game}-s{t.seed}.npz")
    )
    if cache.is_file():
        with np.load(cache) as stored:
            batch = Batch(**{name: stored[name] for name in stored.files})
    else:
        batch = build_validation_set(shard, positions, d.validation_positions_per_game, t.seed)
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp.npz")
        np.savez(temporary, **{f: getattr(batch, f) for f in Batch.__dataclass_fields__})
        os.replace(temporary, cache)
    size = micro_batch(config)
    return [
        _batch_to_tensors(take(batch, np.arange(start, start + size)))
        for start in range(0, positions, size)
    ]


def _loader(dataset: ShardDataset, workers: int) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


def lr_factor(step: int, config: RunConfig) -> float:
    t = config.train
    if step < t.warmup_steps:
        return (step + 1) / t.warmup_steps
    factor = 1.0
    for boundary, value in zip(t.lr_boundaries, t.lr_factors, strict=True):
        if step >= boundary:
            factor = value
    if t.rewarm_start <= step < t.rewarm_start + t.rewarm_steps:
        progress = (step - t.rewarm_start + 1) / t.rewarm_steps
        factor = t.rewarm_from + (factor - t.rewarm_from) * progress
    return factor


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        # Norm gains, biases, embeddings-of-position and soft-MoE scale are not decayed.
        if parameter.ndim < 2 or name.endswith("position"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _atomic_save(state: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def sync_balance_buffers(source: nn.Module, target: nn.Module) -> None:
    """Copy routing balance biases from the live model into the SWA model."""
    live = [m for m in source.modules() if isinstance(m, RoutedMoe)]
    averaged = [m for m in target.modules() if isinstance(m, RoutedMoe)]
    for a, b in zip(live, averaged, strict=True):
        b.balance_bias.copy_(a.balance_bias)


@torch.no_grad()
def calibrate_balance(
    model: nn.Module,
    batches: Iterator[dict[str, torch.Tensor]],
    count: int,
    device: torch.device,
    autocast: bool,
) -> None:
    """Re-tune the routing balance biases for ``model``'s own (SWA-averaged) router.

    The live biases were tuned against the live router weights, and the averaged
    router routes differently (measured on ab_m1: worst expert 1.23x its fair share
    live, 1.49x on the SWA weights with the live biases). Weights stay frozen: each
    batch only counts expert loads and applies one bias update.
    """
    layers = [m for m in model.modules() if isinstance(m, RoutedMoe)]
    if not layers:
        return
    model.train()  # train mode makes the MoE layers count expert loads
    for _ in range(count):
        batch = _to_device(next(batches), device)
        planes = expand_planes(batch["packed"], batch["aux"])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
            model(planes)
        for layer in layers:
            layer.update_balance()
    model.eval()


# Config fields a resumed run may change: they decide how many processes stream the
# data, not what the model, optimizer or data distribution is. Resume already
# reshuffles the stream, so changing them only costs or saves memory.
RESUME_MUTABLE_DATA = frozenset({"workers"})
# Data settings added after runs already existed; a checkpoint saved before them used
# exactly these values.
LATER_DATA_DEFAULTS = {
    "teacher_weight": 1.0,
    "teacher_temperature": 1.0,
    "mirror": False,
    "opening_plies": 0,
    "opening_keep": 1.0,
    "game_directory": "",
    "game_shards": (),
    "game_teacher_labels": "",
}
LATER_TRAIN_DEFAULTS = {"rewarm_start": 0, "rewarm_steps": 0, "rewarm_from": 1.0}
# What a fork may change: the targets, how positions are presented, and the game-target
# shards (a distillation stage after a mixed run drops them).
FORK_MUTABLE_DATA = frozenset(
    {
        "teacher_labels",
        "teacher_weight",
        "teacher_temperature",
        "mirror",
        "opening_plies",
        "opening_keep",
        "game_directory",
        "game_shards",
        "game_teacher_labels",
    }
)
# ... and the schedule it continues on. Optimizer settings, batch and seed stay fixed.
FORK_MUTABLE_TRAIN = frozenset(
    {
        "steps",
        "lr_boundaries",
        "lr_factors",
        "rewarm_start",
        "rewarm_steps",
        "rewarm_from",
        "swa_start",
        "swa_every",
        "validate_every",
        "checkpoint_every",
        "log_every",
        "balance_calibration_batches",
        "compile",
    }
)


def _resume_identity(raw: dict[str, Any]) -> dict[str, Any]:
    data = {**LATER_DATA_DEFAULTS, **raw["data"]}
    data = {k: v for k, v in data.items() if k not in RESUME_MUTABLE_DATA}
    identity = {**raw, "data": data}
    if "train" in raw:
        identity["train"] = {**LATER_TRAIN_DEFAULTS, **raw["train"]}
    return identity


def _fork_identity(raw: dict[str, Any]) -> dict[str, Any]:
    """What a fork must share with its parent: everything but the name, the targets,
    the presentation of positions, the game-target shards and the schedule.

    A fork continues another run's checkpoint, optimizer state included, as a new run:
    e.g. a pure-teacher stage after a mixed run, on a schedule of its own.
    """
    identity = _resume_identity(raw)
    data = {k: v for k, v in identity["data"].items() if k not in FORK_MUTABLE_DATA}
    identity = {**identity, "name": None, "data": data}
    if "train" in identity:
        train = identity["train"]
        identity["train"] = {k: v for k, v in train.items() if k not in FORK_MUTABLE_TRAIN}
    return identity


def initial_weights(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    """The weights an ``--init-from`` run starts at: the checkpoint's SWA average when it
    has one (the net that was exported and played), else its live weights.

    Without end-of-run calibration the SWA routing biases were never tuned, so the live
    ones are kept, as ``export_onnx.load_model`` does.
    """
    live: dict[str, torch.Tensor] = state["model"]
    if not state["swa_started"]:
        return dict(live)
    averaged = {
        key.removeprefix("module."): value
        for key, value in state["swa"].items()
        if key.startswith("module.")
    }
    if not state.get("swa_calibrated", False):
        averaged.update({key: value for key, value in live.items() if key.endswith("balance_bias")})
    return averaged


def compiled_for_training(model: MouseMoe) -> nn.Module:
    """``torch.compile`` of the model for the training step (shares its parameters).

    ab_m1, batch 1024, RTX 5070 (2026-10-25): forward+backward 265 ms eager, 122 ms
    compiled. Validation, calibration and export keep using the eager model.
    """
    import torch._inductor.config as inductor_config

    # Descriptive kernel names overflow Windows' 260-character paths in the Inductor
    # cache (FileNotFoundError on a fused layer-norm kernel's 150-character name).
    # False gives numbered kernel names; the stub only lists the descriptive choices.
    inductor_config.triton.descriptive_names = False  # type: ignore[assignment]
    return cast(nn.Module, torch.compile(model))


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _expert_load_cv(model: MouseMoe) -> float | None:
    """Coefficient of variation of expert load over the last step (0 = perfectly even)."""
    values = []
    for module in model.moe_layers():
        if isinstance(module, RoutedMoe) and module.load.sum() > 0:
            load = module.load.float()
            values.append(float(load.std() / load.mean()))
    return sum(values) / len(values) if values else None


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    config: RunConfig,
    device: torch.device,
    batches: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    seen = 0
    for raw in loader:
        batch = _to_device(raw, device)
        planes = expand_planes(batch["packed"], batch["aux"])
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=config.train.precision == "bf16"
        ):
            outputs = model(planes)
        terms = compute_losses(outputs, batch, config.loss, planes.new_zeros(()))
        values = {f"val_{k}": v for k, v in terms.as_floats().items() if k != "moe_aux"}
        values.update({f"val_{k}": v for k, v in metrics(outputs, batch).items()})
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        seen += 1
        if seen >= batches:
            break
    model.train()
    if seen == 0:
        raise RuntimeError("validation produced no batches; check data.validation_shards")
    return {key: value / seen for key, value in totals.items()}


def train(
    config: RunConfig,
    run_dir: Path,
    device: torch.device,
    resume: bool,
    max_steps: int | None,
    guard: MemoryGuard | None = None,
    fork_from: Path | None = None,
    init_from: Path | None = None,
) -> None:
    t = config.train
    data_dir = (ROOT / config.data.directory).resolve()
    train_shards = verified_shards(
        data_dir, list(config.data.train_shards), config.data.verify_hashes
    )
    val_shards = verified_shards(
        data_dir, list(config.data.validation_shards), config.data.verify_hashes
    )
    if config.data.teacher_labels:
        require_labels(ROOT / config.data.teacher_labels, train_shards)
    extra_shards = game_shards(config)
    if config.data.game_teacher_labels:
        require_labels(ROOT / config.data.game_teacher_labels, extra_shards)
    train_shards += extra_shards
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest = checkpoints / "latest.pt"
    if latest.exists() and not resume:
        raise FileExistsError(f"{latest} exists; pass --resume or choose a new run name")
    if sum([resume, fork_from is not None, init_from is not None]) > 1:
        raise ValueError("--resume, --fork-from and --init-from are exclusive")

    torch.manual_seed(t.seed)
    random.seed(t.seed)
    np.random.seed(t.seed)
    model = MouseMoe(config.model).to(device)
    count = parameter_count(model)
    if count > config.model.parameter_budget:
        raise ValueError(
            f"model has {count:,} parameters, over budget {config.model.parameter_budget:,}"
        )
    optimizer = torch.optim.NAdam(
        _param_groups(model, t.weight_decay),
        lr=t.lr,
        betas=t.betas,
        eps=t.eps,
        decoupled_weight_decay=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, config))
    swa = AveragedModel(model)
    swa_started = False
    swa_calibrated = False
    step = 0
    if resume or fork_from is not None:
        source = latest if resume or fork_from is None else fork_from
        if not source.exists():
            raise FileNotFoundError(f"checkpoint {source} does not exist")
        # Load to the CPU: load_state_dict copies into the device tensors (the optimizer
        # moves its state to each parameter's device). Loading to the GPU kept a second
        # copy of model, optimizer and SWA state resident for the whole run, and resuming
        # ab_m1_fixed ran out of the 12 GB card's memory on its first step.
        state = torch.load(source, map_location="cpu", weights_only=False)
        identity = _resume_identity if resume else _fork_identity
        if identity(state["config"]) != identity(to_dict(config)):
            raise ValueError("checkpoint config differs from --config; refusing to continue it")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        swa.load_state_dict(state["swa"])
        swa_started = state["swa_started"]
        swa_calibrated = state.get("swa_calibrated", False)
        step = state["step"]
        parent_swa_start = state["config"]["train"]["swa_start"]
        if fork_from is not None and swa_started and parent_swa_start < t.swa_start:
            # The parent averaged from an earlier step than this run's window starts.
            if step >= t.swa_start:
                raise ValueError(
                    f"{source} holds SWA snapshots from step {parent_swa_start}, before this "
                    f"run's swa_start {t.swa_start}; they cannot be separated out"
                )
            swa = AveragedModel(model)
            swa_started = swa_calibrated = False
        del state
    if init_from is not None:
        # Weights only: optimizer, schedule, SWA and step start fresh, and the data may
        # differ (e.g. a game-target run fine-tuned on teacher labels).
        state = torch.load(init_from, map_location="cpu", weights_only=False)
        if state["config"]["model"] != to_dict(config)["model"]:
            raise ValueError(f"{init_from} has a different model config; cannot start from it")
        model.load_state_dict(initial_weights(state))
        source_step = state["step"]
        del state
    (run_dir / "config.json").write_text(
        json.dumps(to_dict(config), indent=2) + "\n", encoding="utf-8"
    )

    train_loader = _loader(
        ShardDataset(train_shards, config, t.seed + step, True), config.data.workers
    )
    val_batches = validation_batches(config, val_shards[0])
    target_steps = t.steps if max_steps is None else min(t.steps, step + max_steps)
    log = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")

    def emit(record: dict[str, Any]) -> None:
        record["time"] = time.time()
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    if fork_from is not None:
        emit({"event": "forked", "from": str(fork_from), "step": step})
    if init_from is not None:
        emit({"event": "initialized", "from": str(init_from), "source_step": source_step})

    def save(path: Path) -> None:
        _atomic_save(
            {
                "config": to_dict(config),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "swa": swa.state_dict(),
                "swa_started": swa_started,
                "swa_calibrated": swa_calibrated,
                "step": step,
                "parameters": count,
                "torch": torch.__version__,
            },
            path,
        )

    emit(
        {
            "event": "start",
            "step": step,
            "parameters": count,
            "device": str(device),
            "train_shards": len(train_shards),
            "validation_shards": len(val_shards),
        }
    )
    model.train()
    step_model = compiled_for_training(model) if t.compile else model
    window_start, window_positions = time.perf_counter(), 0
    autocast = t.precision == "bf16"
    accum = t.grad_accum_steps
    batches = iter(train_loader)
    while step < target_steps:
        if guard is not None:
            try:
                guard.check()
            except MemoryReserveError:
                # Keep the work done so far; resume continues from here.
                save(latest)
                emit({"event": "memory_stop", "step": step})
                log.close()
                raise
        optimizer.zero_grad(set_to_none=True)
        # Expert loads accumulate over the micro-batches, so the balance biases take
        # one update per optimizer step, as with one full-size batch.
        for _ in range(accum):
            batch = _to_device(next(batches), device)
            planes = expand_planes(batch["packed"], batch["aux"])
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                outputs = step_model(planes)
                terms = compute_losses(outputs, batch, config.loss, model.aux_loss())
            (terms.total / accum).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        if not math.isfinite(float(grad_norm)):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        scheduler.step()
        load_cv = _expert_load_cv(model) if (step + 1) % t.log_every == 0 else None
        model.update_balance()
        step += 1
        window_positions += t.batch_size
        if step >= t.swa_start and step % t.swa_every == 0:
            swa.update_parameters(model)
            swa_started = True
        if step % t.log_every == 0:
            elapsed = time.perf_counter() - window_start
            record = {
                "event": "train",
                "step": step,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "positions_per_s": window_positions / elapsed,
            }
            record.update(terms.as_floats())
            record.update(metrics(outputs, batch))
            if load_cv is not None:
                record["expert_load_cv"] = load_cv
            emit(record)
            window_start, window_positions = time.perf_counter(), 0
        if step == t.steps and swa_started and t.balance_calibration_batches:
            sync_balance_buffers(model, swa.module)
            calibrate_balance(swa.module, batches, t.balance_calibration_batches, device, autocast)
            swa_calibrated = True
            emit({"event": "balance_calibrated", "step": step})
        if step % t.validate_every == 0 or step == target_steps:
            evaluated: nn.Module = model
            if swa_started:
                if not swa_calibrated:
                    sync_balance_buffers(model, swa.module)
                evaluated = swa.module
            result = validate(evaluated, val_batches, config, device, len(val_batches))
            emit({"event": "validation", "step": step, "swa": swa_started, **result})
        if step % t.checkpoint_every == 0 or step == target_steps:
            save(checkpoints / f"step_{step:07d}.pt")
            save(latest)
    emit({"event": "stop", "step": step})
    log.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a Mouse-MoE network")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--resume", action="store_true", help="continue from runs/<name>/checkpoints/latest.pt"
    )
    parser.add_argument(
        "--fork-from",
        type=Path,
        help="continue another run's checkpoint, optimizer included, as a new run "
        "(same model; targets and schedule may change)",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        help="start a fresh run (new schedule and data) from a checkpoint's weights",
    )
    parser.add_argument(
        "--max-steps", type=int, help="stop after this many more steps (smoke tests)"
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--reserve-ram-gib",
        type=float,
        default=4.0,
        help="system RAM kept free for other programs (hard job limit and watchdog)",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.92,
        help="cap on PyTorch GPU memory, so the driver never spills into system RAM",
    )
    args = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("run with python -I: training data is untrusted input")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is not available; refusing to fall back to CPU (pass --device cpu explicitly)"
        )
    config = load_config(args.config)
    device = torch.device(args.device)
    guard = None
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        guard = MemoryGuard(args.reserve_ram_gib, args.gpu_memory_fraction, device)
        print(guard.describe(), flush=True)
    try:
        with no_idle_sleep() if args.device == "cuda" else contextlib.nullcontext():
            train(
                config,
                args.runs / config.name,
                device,
                args.resume,
                args.max_steps,
                guard,
                args.fork_from,
                args.init_from,
            )
    finally:
        if guard is not None:
            guard.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
