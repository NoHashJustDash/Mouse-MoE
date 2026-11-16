# SPDX-License-Identifier: GPL-3.0-or-later
"""Label training shards with a teacher network (Lc0 BT4) for distillation.

    python -I -m mouse_moe.label_teacher --onnx <BT4 ONNX from lc0 leela2onnx> `
        --config configs\\distill\\ab_kd_game.toml --out data\\test80\\labels\\bt4-it332

The teacher ONNX comes from ``lc0 leela2onnx`` (input ``/input/planes``, outputs
``/output/policy``, ``/output/wdl``, ``/output/mlh``). It runs as a TensorRT FP16 engine,
cached next to the labels. Each shard is labelled whole, in archive order, and written
with ``mouse_moe.teacher.write_shard_labels``. Finished shards are skipped, so a stopped
run resumes. Per shard it reports throughput, the share of positions where the
teacher's top move is the data's most-visited move (an alignment check: about 68% for
BT4-it332), and the mean policy mass kept by the top-K moves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from mouse_moe.config import RunConfig, load_config
from mouse_moe.model import expand_planes
from mouse_moe.shards import iter_games, verified_shards
from mouse_moe.teacher import TOP_K, write_shard_labels
from mouse_moe.train import ROOT
from mouse_moe.v6 import parse_records, records_to_batch

# The TensorRT SDK's lib folder. Use the SDK whose Python wheel is installed (10.9.0.34, the
# version the Mouse engine is built against); its DLLs are loaded from there.
TENSORRT_LIB_ENV = "MOUSE_MOE_TENSORRT_LIB"


def tensorrt_lib() -> Path:
    """The TensorRT lib folder named by ``MOUSE_MOE_TENSORRT_LIB``; fails if unset or missing."""
    value = os.environ.get(TENSORRT_LIB_ENV)
    if not value:
        raise RuntimeError(
            f"set {TENSORRT_LIB_ENV} to the lib folder of the TensorRT SDK whose Python wheel "
            "is installed (TensorRT-10.9.0.34\lib)"
        )
    path = Path(value)
    if not path.is_dir():
        raise RuntimeError(f"{TENSORRT_LIB_ENV} is not a folder: {path}")
    return path

ENGINE_BATCH = 512  # TensorRT 10.9 FP16 on the RTX 5070: ~4,700-5,000 positions/s at 64-1024
CHUNK = 8192  # positions parsed on the CPU per hand-off to the GPU

# planes [N, 112, 8, 8] -> (policy logits [N, 1858], WDL probabilities [N, 3], MLH [N, 1])
Evaluate = Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]


def select_top_moves(logits: Tensor, legal: Tensor, k: int = TOP_K) -> tuple[Tensor, Tensor]:
    """The k most likely legal moves and their probabilities (softmax over legal moves).

    Positions with fewer than k legal moves get probability 0 in the extra columns;
    ``ShardLabels.apply`` ignores those.
    """
    probs = logits.float().masked_fill(~legal, float("-inf")).softmax(dim=-1)
    top = probs.topk(k, dim=-1)
    return top.indices, top.values


@dataclass
class _Chunk:
    names: list[str]
    lengths: list[int]
    packed: np.ndarray  # type: ignore[type-arg]
    aux: np.ndarray  # type: ignore[type-arg]
    policy: np.ndarray  # type: ignore[type-arg]


def _chunks(shard: Path) -> Iterator[_Chunk]:
    names: list[str] = []
    lengths: list[int] = []
    parts: list = []  # type: ignore[type-arg]
    count = 0
    for name, raw in iter_games(shard):
        game = records_to_batch(parse_records(raw))
        names.append(name)
        lengths.append(len(game))
        parts.append(game)
        count += len(game)
        if count >= CHUNK:
            yield _Chunk(names, lengths, *_stack(parts))
            names, lengths, parts, count = [], [], [], 0
    if parts:
        yield _Chunk(names, lengths, *_stack(parts))


def _stack(parts: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    return (
        np.concatenate([p.packed for p in parts]),
        np.concatenate([p.aux for p in parts]),
        np.concatenate([p.policy for p in parts]),
    )


def _prefetch(iterator: Iterator[_Chunk], depth: int = 3) -> Iterator[_Chunk]:
    """Parse the next chunks on a thread while the GPU labels the current one."""
    slots: queue.Queue[_Chunk | BaseException | None] = queue.Queue(maxsize=depth)

    def produce() -> None:
        try:
            for item in iterator:
                slots.put(item)
            slots.put(None)
        except BaseException as error:  # re-raised in the consumer, never swallowed
            slots.put(error)

    threading.Thread(target=produce, daemon=True).start()
    while True:
        item = slots.get()
        if item is None:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def label_shard(
    shard: Path, evaluate: Evaluate, device: torch.device, out_dir: Path, teacher_sha256: str
) -> dict[str, float]:
    """Label every position of ``shard`` and write its label files; returns statistics."""
    started = time.perf_counter()
    names: list[str] = []
    lengths: list[int] = []
    outputs: dict[str, list[np.ndarray]] = {"idx": [], "prob": [], "wdl": [], "mlh": []}  # type: ignore[type-arg]
    agree = mass = 0.0
    for chunk in _prefetch(_chunks(shard)):
        names += chunk.names
        lengths += chunk.lengths
        for start in range(0, len(chunk.packed), ENGINE_BATCH):
            end = start + ENGINE_BATCH
            packed = torch.from_numpy(chunk.packed[start:end]).to(device)
            aux = torch.from_numpy(chunk.aux[start:end]).to(device)
            target = torch.from_numpy(chunk.policy[start:end]).to(device)
            legal = target >= 0
            with torch.no_grad():
                logits, wdl, mlh = evaluate(expand_planes(packed, aux))
                idx, prob = select_top_moves(logits, legal)
            agree += float((idx[:, 0] == target.argmax(dim=1)).sum())
            mass += float(prob.sum())
            outputs["idx"].append(idx.to(torch.int32).cpu().numpy().astype(np.uint16))
            outputs["prob"].append(prob.half().cpu().numpy())
            outputs["wdl"].append(wdl.half().cpu().numpy())
            outputs["mlh"].append(mlh.half().reshape(-1).cpu().numpy())
    rows = int(sum(lengths))
    write_shard_labels(
        out_dir,
        shard.name,
        names,
        lengths,
        {key: np.concatenate(value) for key, value in outputs.items()},
        teacher_sha256,
    )
    seconds = time.perf_counter() - started
    return {
        "positions": rows,
        "games": len(names),
        "seconds": round(seconds, 1),
        "positions_per_s": round(rows / seconds),
        "top1_matches_data": round(agree / rows, 4),
        "top_k_mass": round(mass / rows, 4),
    }


class TensorRTTeacher:
    """An FP16 TensorRT engine for the teacher ONNX, fed and read as torch CUDA tensors."""

    def __init__(self, onnx_path: Path, cache_dir: Path) -> None:
        lib = tensorrt_lib()
        os.environ["PATH"] = str(lib) + os.pathsep + os.environ["PATH"]
        os.add_dll_directory(str(lib))
        import tensorrt as trt  # type: ignore[import-untyped]  # the wheel ships no stubs

        self.trt = trt
        digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        self.sha256 = digest
        plan = cache_dir / f"teacher-{digest[:16]}-trt{trt.__version__}-b{ENGINE_BATCH}.plan"
        logger = trt.Logger(trt.Logger.WARNING)
        if not plan.is_file():
            cache_dir.mkdir(parents=True, exist_ok=True)
            builder = trt.Builder(logger)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(str(onnx_path)):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(f"TensorRT could not parse {onnx_path}: {errors}")
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)
            profile = builder.create_optimization_profile()
            name = network.get_input(0).name
            profile.set_shape(
                name, (1, 112, 8, 8), (ENGINE_BATCH, 112, 8, 8), (ENGINE_BATCH, 112, 8, 8)
            )
            config.add_optimization_profile(profile)
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("TensorRT engine build failed")
            temporary = plan.with_name(plan.name + ".tmp")
            temporary.write_bytes(bytes(serialized))
            os.replace(temporary, plan)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(plan.read_bytes())
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input = next(
            n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        )
        self.outputs = {key: next(n for n in names if key in n) for key in ("policy", "wdl", "mlh")}
        self.dtype = (
            torch.float16
            if self.engine.get_tensor_dtype(self.input) == trt.float16
            else torch.float32
        )

    def __call__(self, planes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch = planes.shape[0]
        planes = planes.to(self.dtype).contiguous()
        self.context.set_input_shape(self.input, tuple(planes.shape))
        self.context.set_tensor_address(self.input, planes.data_ptr())
        results = {}
        for key, name in self.outputs.items():
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = (
                torch.float16
                if self.engine.get_tensor_dtype(name) == self.trt.float16
                else torch.float32
            )
            results[key] = torch.empty(shape, dtype=dtype, device=planes.device)
            self.context.set_tensor_address(name, results[key].data_ptr())
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT inference failed")
        stream.synchronize()
        if results["policy"].shape[0] != batch:
            raise RuntimeError("TensorRT returned the wrong batch size")
        return results["policy"], results["wdl"], results["mlh"]


def shards_to_label(config: RunConfig, include_validation: bool) -> list[Path]:
    """The config's train shards, after its validation shard when asked for.

    The validation shard is small and goes first, so its alignment figures can be checked
    before the hours of training shards.
    """
    d = config.data
    names = [*d.validation_shards, *d.train_shards] if include_validation else [*d.train_shards]
    return verified_shards((ROOT / d.directory).resolve(), names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label training shards with a teacher net")
    parser.add_argument("--onnx", type=Path, required=True, help="teacher from lc0 leela2onnx")
    parser.add_argument("--config", type=Path, required=True, help="labels its train shards")
    parser.add_argument("--out", type=Path, required=True, help="label directory")
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="also label the validation shard, first, for mouse_moe.score_teacher",
    )
    parser.add_argument("--reserve-ram-gib", type=float, default=4.0)
    args = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("run with python -I: training data is untrusted input")
    if not torch.cuda.is_available():
        parser.error("CUDA is not available; the teacher runs on the GPU only")
    from mouse_moe.resources import MemoryGuard, no_idle_sleep

    shards = shards_to_label(load_config(args.config), args.include_validation)
    device = torch.device("cuda")
    guard = MemoryGuard(args.reserve_ram_gib, 0.92, device)
    print(guard.describe(), flush=True)
    try:
        with no_idle_sleep():
            _label_all(args, shards, device)
    finally:
        guard.close()
    return 0


def _label_all(args: argparse.Namespace, shards: list[Path], device: torch.device) -> None:
    teacher = TensorRTTeacher(args.onnx, args.out)
    for shard in shards:
        done = args.out / f"{shard.name}.done.json"
        if done.is_file() and json.loads(done.read_text())["teacher_sha256"] == teacher.sha256:
            print(json.dumps({"shard": shard.name, "skipped": "already labelled"}), flush=True)
            continue
        stats = label_shard(shard, teacher, device, args.out, teacher.sha256)
        print(json.dumps({"shard": shard.name, **stats}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
