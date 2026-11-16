# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure training throughput, GPU memory and data-loader speed for a config.

    python -I -m mouse_moe.benchmark --config configs/m1_46m.toml

Reports positions/s for the model step alone (real data batch, repeated) and for
the data loader alone, then estimates wall time for the config's full step count.
Run it with nothing else using the GPU, or the numbers are meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from mouse_moe.config import load_config
from mouse_moe.dataset import StreamSpec, stream_batches
from mouse_moe.losses import compute_losses
from mouse_moe.model import MouseMoe, expand_planes, parameter_count
from mouse_moe.resources import MemoryGuard
from mouse_moe.shards import verified_shards
from mouse_moe.train import (
    ROOT,
    ShardDataset,
    _batch_to_tensors,
    _loader,
    _to_device,
    compiled_for_training,
    game_shards,
    micro_batch,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--loader-batches", type=int, default=100)
    parser.add_argument("--reserve-ram-gib", type=float, default=4.0)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.92)
    args = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("run with python -I: training data is untrusted input")
    if not torch.cuda.is_available():
        parser.error("CUDA is not available")
    config = load_config(args.config)
    device = torch.device("cuda")
    guard = MemoryGuard(args.reserve_ram_gib, args.gpu_memory_fraction, device)
    print(guard.describe(), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    data_dir = (ROOT / config.data.directory).resolve()
    train_shards = verified_shards(data_dir, list(config.data.train_shards)) + game_shards(config)

    spec = StreamSpec(
        shards=(train_shards[0],),
        batch_size=micro_batch(config),
        shuffle_buffer=micro_batch(config),
        sample_rate=1.0,
        readers=1,
        cycle=True,
        seed=0,
    )
    batch = _to_device(_batch_to_tensors(next(stream_batches(spec))), device)

    model = MouseMoe(config.model).to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=1e-4)
    step_model = compiled_for_training(model) if config.train.compile else model
    torch.cuda.reset_peak_memory_stats()
    bf16 = config.train.precision == "bf16"
    timings = []
    for step in range(args.warmup + args.steps):
        guard.check()
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.train.grad_accum_steps):  # one optimizer step's micro-batches
            planes = expand_planes(batch["packed"], batch["aux"])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                terms = compute_losses(step_model(planes), batch, config.loss, model.aux_loss())
            (terms.total / config.train.grad_accum_steps).backward()
        optimizer.step()
        model.update_balance()
        torch.cuda.synchronize()
        if step >= args.warmup:
            timings.append(time.perf_counter() - start)
    step_seconds = sorted(timings)[len(timings) // 2]
    model_rate = config.train.batch_size / step_seconds

    loader = _loader(ShardDataset(train_shards, config, 0, True), config.data.workers)
    iterator = iter(loader)
    next(iterator)  # workers fill their shuffle buffers first
    start = time.perf_counter()
    for _ in range(args.loader_batches):
        next(iterator)
    elapsed = time.perf_counter() - start
    loader_rate = args.loader_batches * micro_batch(config) / elapsed

    rate = min(model_rate, loader_rate)
    report = {
        "config": config.name,
        "parameters": parameter_count(model),
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": config.train.batch_size,
        "grad_accum_steps": config.train.grad_accum_steps,
        "compile": config.train.compile,
        "model_positions_per_s": round(model_rate),
        "loader_positions_per_s": round(loader_rate),
        "bottleneck": "model" if model_rate <= loader_rate else "data loader",
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "estimated_hours_for_config_steps": round(
            config.train.steps * config.train.batch_size / rate / 3600, 1
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
