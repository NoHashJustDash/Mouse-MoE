# SPDX-License-Identifier: GPL-3.0-or-later
"""Score checkpoints against teacher labels on a held-out shard.

    python -I -m mouse_moe.score_teacher --config configs\\m20\\stage3.toml `
        --checkpoint runs\\m20-stage2\\checkpoints\\step_0750000.pt `
        --checkpoint runs\\m20-stage3\\checkpoints\\step_0100000.pt

Training validation always scores against the games' own targets, so it cannot show how
close a distilled net has come to its teacher. This scores each checkpoint's played net
(its SWA average once started, see ``train.initial_weights``) on the same positions of the
config's validation shard, which needs teacher labels of its own
(``label_teacher --include-validation``). Each checkpoint uses its saved model and loss
settings, so nets from different runs compare directly. One JSON line per checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mouse_moe.config import ConfigError, RunConfig, load_config, load_config_dict
from mouse_moe.dataset import build_validation_set
from mouse_moe.model import MouseMoe
from mouse_moe.shards import verified_shards
from mouse_moe.teacher import ShardLabels
from mouse_moe.train import (
    ROOT,
    _batch_to_tensors,
    initial_weights,
    micro_batch,
    validate,
)
from mouse_moe.v6 import take


def teacher_validation_batches(config: RunConfig) -> list[dict[str, torch.Tensor]]:
    """A fixed sample of the validation shard, with the teacher's targets."""
    d, t = config.data, config.train
    if not d.teacher_labels:
        raise ConfigError("scoring against a teacher needs data.teacher_labels")
    shard = verified_shards((ROOT / d.directory).resolve(), list(d.validation_shards))[0]
    labels = ShardLabels.open(ROOT / d.teacher_labels, shard.name)
    positions = t.validation_batches * t.batch_size
    sample = build_validation_set(
        shard, positions, d.validation_positions_per_game, t.seed, teacher=labels
    )
    size = micro_batch(config)
    return [
        _batch_to_tensors(take(sample, np.arange(start, start + size)))
        for start in range(0, positions, size)
    ]


def score_checkpoint(
    path: Path, batches: list[dict[str, torch.Tensor]], device: torch.device
) -> dict[str, Any]:
    """The losses of a checkpoint's played net on ``batches``."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = load_config_dict(state["config"])
    model = MouseMoe(config.model).to(device)
    model.load_state_dict(initial_weights(state))
    result = validate(model, batches, config, device, len(batches))
    return {"checkpoint": str(path), "step": state["step"], "swa": state["swa_started"], **result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score checkpoints against teacher labels")
    parser.add_argument("--config", type=Path, required=True, help="names shard and labels")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("run with python -I: training data is untrusted input")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is not available; refusing to fall back to CPU (pass --device cpu explicitly)"
        )
    device = torch.device(args.device)
    batches = teacher_validation_batches(load_config(args.config))
    for path in args.checkpoint:
        print(json.dumps(score_checkpoint(path, batches, device)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
