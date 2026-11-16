# SPDX-License-Identifier: GPL-3.0-or-later
"""Export a trained checkpoint to the Mouse engine's ONNX contract and verify it.

    python -I -m mouse_moe.export_onnx --checkpoint runs/<name>/checkpoints/latest.pt \
        --output runs/<name>/mouse-moe.onnx

Contract (``engine/nn/model_metadata.cpp``): opset 17, IR version 8, standard
domain only; input ``planes`` [batch,112,8,8]; outputs ``policy`` [batch,1858]
logits, ``wdl`` [batch,3] probabilities, ``mlh`` [batch,1]; dynamic batch axis
named ``batch``; ``mouse.*`` metadata. After export the file is checked with
``onnx.checker`` and ONNX Runtime (CPU) against PyTorch on a real validation
batch. Any mismatch deletes the output and fails.

``--precision fp16`` exports a float16 graph for engine backends that run the graph's
own precision (CUDA, DirectML): weights and arithmetic in float16, the engine's
float32 input and outputs cast at the edges, and the routers kept in float32 so expert
choice does not depend on half-precision rounding. It is compared with the float32
PyTorch model, within the FP16 bounds below.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
import torch

from mouse_moe import __version__
from mouse_moe.config import RunConfig, load_config_dict
from mouse_moe.model import (
    EngineWrapper,
    HalfEngineWrapper,
    MouseMoe,
    RoutedMoe,
    parameter_count,
)
from mouse_moe.shards import iter_games, verified_shards
from mouse_moe.train import ROOT, sync_balance_buffers
from mouse_moe.v6 import concat_batches, expand_planes, parse_records, records_to_batch

OPSET = 17
IR_VERSION = 8
# The engine requires a positive integer and treats it as provenance only.
MOUSE_MOE_NETWORK_FORMAT = "1001"
# FP32 PyTorch vs FP32 ONNX Runtime; policy logits can reach |20|, so these absolute
# bounds sit well above accumulated rounding and far below anything search-visible
# (the engine's own FP16 parity gate is 0.005 on probabilities).
TOLERANCE = {"policy": 1e-3, "wdl": 1e-4, "mlh": 1e-2}
# Boards where some router's k-th and (k+1)-th biased scores are this close are decided
# by rounding: one float32 step at the scores' scale (~1) is 1.2e-7, and ONNX Runtime
# rounds the router differently from PyTorch. ab_kd_m1t60 (2026-10-25, 4,096 boards):
# the 4 boards over the policy bound had gaps under 3e-7, while all 4,076 boards with
# gaps of 3e-7 or more matched to 3e-5. The ONNX Runtime comparison skips boards under
# the margin (55 there, 1.3%), and the export fails if they exceed MAX_TIED_SHARE.
ROUTING_TIE_MARGIN = 1e-6
MAX_TIED_SHARE = 0.05
# FP16 against PyTorch FP32 on decided boards, over all 1858 policy logits; also the
# TensorRT expert plugin's bounds (export_plugin.py). m1 (2026-10-27): the plugin on 512
# validation positions, top-1 98.4%, mean KL 2.5e-4, max WDL error 0.016; DirectML FP16
# on 4,096 held-out test80 positions (legal moves only), 98.7%, 2.2e-4 and 0.107.
FP16_MIN_TOP1 = 0.97
FP16_MAX_MEAN_KL = 5e-4
FP16_MAX_WDL_ERROR = 0.15


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 22):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint: Path, use_swa: bool) -> tuple[MouseMoe, RunConfig]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = load_config_dict(state["config"])
    model = MouseMoe(config.model)
    model.load_state_dict(state["model"])
    if use_swa:
        if not state["swa_started"]:
            raise ValueError("checkpoint has no SWA average yet; pass --weights raw explicitly")
        averaged = MouseMoe(config.model)
        swa_state = {
            key.removeprefix("module."): value
            for key, value in state["swa"].items()
            if key.startswith("module.")
        }
        averaged.load_state_dict(swa_state)
        if not state.get("swa_calibrated", False):
            # Checkpoints from before end-of-run calibration: the live biases are the
            # best available, but they were tuned for the live router.
            print("note: SWA routing biases were not calibrated; using the live biases")
            sync_balance_buffers(model, averaged)
        model = averaged
    return model.eval(), config


def validation_planes(config: RunConfig, positions: int) -> np.ndarray:  # type: ignore[type-arg]
    data_dir = (ROOT / config.data.directory).resolve()
    shard = verified_shards(data_dir, [config.data.validation_shards[0]])[0]
    batches = []
    total = 0
    for _, raw in iter_games(shard):
        batch = records_to_batch(parse_records(raw))
        batches.append(batch)
        total += len(batch)
        if total >= positions:
            break
    merged = concat_batches(batches)
    return expand_planes(merged.packed[:positions], merged.aux[:positions])


def export(
    checkpoint: Path, output: Path, use_swa: bool, positions: int, precision: str = "fp32"
) -> dict[str, float]:
    model, config = load_model(checkpoint, use_swa)
    reference_planes = torch.from_numpy(validation_planes(config, positions))
    return export_model(model, reference_planes, output, _sha256(checkpoint), precision)


def half_model(model: MouseMoe) -> MouseMoe:
    """A float16 copy of ``model`` whose routers and balance biases stay float32."""
    half = copy.deepcopy(model).half()
    for layer in half.moe_layers():
        if isinstance(layer, RoutedMoe):
            layer.router.float()
            layer.balance_bias = layer.balance_bias.float()
    return half.eval()


def fp16_agreement(
    expected: tuple[torch.Tensor, ...],
    actual: list[npt.NDArray[np.float32]],
    decided: npt.NDArray[np.bool_],
) -> dict[str, float]:
    """Top-1 agreement, mean policy KL and max WDL error of an FP16 graph vs FP32."""
    reference = expected[0].double()[decided].log_softmax(1)
    fp16 = torch.from_numpy(actual[0]).double()[decided].log_softmax(1)
    return {
        "top1_agreement": float((reference.argmax(1) == fp16.argmax(1)).double().mean()),
        "mean_policy_kl": float((reference.exp() * (reference - fp16)).sum(1).mean()),
        "max_wdl_error": float(np.abs(expected[1].numpy()[decided] - actual[1][decided]).max()),
    }


def dispatch_agrees(training: torch.Tensor, exported: torch.Tensor) -> bool:
    """Whether the export dispatch reproduces the training dispatch up to rounding.

    The static export sums all experts in a different order from the training
    dispatch. float32 rounding is relative, and moves-left reaches ~230 plies
    (ab_m1: 6.9e-5 absolute, 3e-7 relative), so the bound has a relative term.
    """
    return torch.allclose(training, exported, atol=1e-5, rtol=1e-5)


def routing_margins(model: MouseMoe, planes: torch.Tensor) -> torch.Tensor:
    """Per board, the smallest routing margin over every token-routed layer and square."""
    margins: list[torch.Tensor] = []
    layers = [m for m in model.moe_layers() if isinstance(m, RoutedMoe)]
    hooks = [
        layer.register_forward_pre_hook(
            lambda module, args: margins.append(module.routing_margin(args[0]))
        )
        for layer in layers
    ]
    try:
        with torch.no_grad():
            EngineWrapper(model)(planes)
    finally:
        for hook in hooks:
            hook.remove()
    if not margins:
        return torch.full((planes.shape[0],), float("inf"))
    return torch.stack(margins).amin(dim=0)


def model_metadata(model: MouseMoe, source_sha256: str, precision: str = "fp32") -> dict[str, str]:
    """The ``mouse.*`` metadata the engine requires (``engine/nn/model_metadata.cpp``)."""
    return {
        "mouse.schema_version": "1",
        "mouse.converter_version": f"mouse-moe-{__version__}",
        "mouse.source_sha256": source_sha256,
        "mouse.input_format": "1",
        "mouse.network_format": MOUSE_MOE_NETWORK_FORMAT,
        "mouse.raw_network_format": MOUSE_MOE_NETWORK_FORMAT,
        "mouse.policy_format": "3",
        "mouse.value_format": "2",
        "mouse.moves_left_format": "1",
        "mouse.policy_mapping": "lc0-1858-v1",
        "mouse.learned_parameter_count": str(parameter_count(model)),
        "mouse.precision": precision,
    }


def stable_softplus(graph: onnx.GraphProto, dtype: int = onnx.TensorProto.FLOAT) -> int:
    """Rewrite every Softplus as relu(x) + log(1 + exp(-|x|)); return how many.

    The same function, but exp never sees a large argument. TensorRT 10.9 FP16 computes
    Softplus as log(1 + exp(x)), which overflows once x passes ~11; moves-left (and Mish
    inputs) go far beyond that, so m1 returned infinite moves-left in FP16.
    """
    one = "stable_softplus_one"
    nodes: list[onnx.NodeProto] = []
    count = 0
    for node in graph.node:
        if node.op_type != "Softplus":
            nodes.append(node)
            continue
        x, prefix = node.input[0], f"{node.name or node.output[0]}/stable"
        nodes += [
            onnx.helper.make_node("Relu", [x], [f"{prefix}/relu"], name=f"{prefix}/Relu"),
            onnx.helper.make_node("Abs", [x], [f"{prefix}/abs"], name=f"{prefix}/Abs"),
            onnx.helper.make_node(
                "Neg", [f"{prefix}/abs"], [f"{prefix}/neg"], name=f"{prefix}/Neg"
            ),
            onnx.helper.make_node(
                "Exp", [f"{prefix}/neg"], [f"{prefix}/exp"], name=f"{prefix}/Exp"
            ),
            onnx.helper.make_node(
                "Add", [f"{prefix}/exp", one], [f"{prefix}/one_plus"], name=f"{prefix}/AddOne"
            ),
            onnx.helper.make_node(
                "Log", [f"{prefix}/one_plus"], [f"{prefix}/log"], name=f"{prefix}/Log"
            ),
            onnx.helper.make_node(
                "Add", [f"{prefix}/relu", f"{prefix}/log"], [node.output[0]], name=f"{prefix}/Add"
            ),
        ]
        count += 1
    if count:
        graph.initializer.append(onnx.helper.make_tensor(one, dtype, [], [1.0]))
        del graph.node[:]
        graph.node.extend(nodes)
    return count


def export_model(
    model: MouseMoe,
    reference_planes: torch.Tensor,
    output: Path,
    source_sha256: str,
    precision: str = "fp32",
) -> dict[str, float]:
    """Export ``model`` and verify ONNX Runtime matches PyTorch on ``reference_planes``."""
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"unsupported export precision: {precision}")
    model.eval()
    with torch.no_grad():
        model.set_export_mode(False)
        expected = EngineWrapper(model)(reference_planes)
        model.set_export_mode(True)
        exported_path = EngineWrapper(model)(reference_planes)
    for name, a, b in zip(("policy", "wdl", "mlh"), expected, exported_path, strict=True):
        if not dispatch_agrees(a, b):
            raise AssertionError(f"export dispatch differs from training dispatch on {name}")
    decided = (routing_margins(model, reference_planes) >= ROUTING_TIE_MARGIN).numpy()
    tied = int((~decided).sum())
    if tied > MAX_TIED_SHARE * len(decided):
        raise AssertionError(
            f"{tied} of {len(decided)} reference boards have routing ties under "
            f"{ROUTING_TIE_MARGIN}; too many to verify the export on the rest"
        )

    temporary = output.with_suffix(".tmp.onnx")
    wrapper = EngineWrapper(model) if precision == "fp32" else HalfEngineWrapper(half_model(model))
    torch.onnx.export(
        wrapper,
        (reference_planes[:2],),
        str(temporary),
        input_names=["planes"],
        output_names=["policy", "wdl", "mlh"],
        dynamic_axes={name: {0: "batch"} for name in ("planes", "policy", "wdl", "mlh")},
        opset_version=OPSET,
        dynamo=False,
    )
    try:
        graph = onnx.load(str(temporary))
        graph.ir_version = IR_VERSION
        domains = {(item.domain, item.version) for item in graph.opset_import}
        if domains != {("", OPSET)}:
            raise AssertionError(f"export used non-standard opsets: {domains}")
        half = precision == "fp16"
        stable_softplus(graph.graph, onnx.TensorProto.FLOAT16 if half else onnx.TensorProto.FLOAT)
        onnx.helper.set_model_props(graph, model_metadata(model, source_sha256, precision))
        onnx.checker.check_model(graph, full_check=True)
        onnx.save(graph, str(temporary))
        session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
        actual = session.run(None, {"planes": reference_planes.numpy()})
        report: dict[str, float] = {"routing_tied_boards": tied}
        if half:
            report |= fp16_agreement(expected, actual, decided)
            if (
                report["top1_agreement"] < FP16_MIN_TOP1
                or report["mean_policy_kl"] > FP16_MAX_MEAN_KL
                or report["max_wdl_error"] > FP16_MAX_WDL_ERROR
                or not all(np.isfinite(value).all() for value in actual)
            ):
                raise AssertionError(f"FP16 export differs from the FP32 model: {report}")
            os.replace(temporary, output)
            return report
        for name, torch_value, ort_value in zip(
            ("policy", "wdl", "mlh"), expected, actual, strict=True
        ):
            error = float(np.abs(torch_value.numpy()[decided] - ort_value[decided]).max())
            report[f"max_abs_error_{name}"] = error
            if error > TOLERANCE[name]:
                raise AssertionError(f"ONNX Runtime {name} differs from PyTorch by {error}")
        os.replace(temporary, output)
        return report
    finally:
        temporary.unlink(missing_ok=True)
        model.set_export_mode(False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a Mouse-MoE checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", choices=["swa", "raw"], default="swa")
    parser.add_argument(
        "--positions", type=int, default=256, help="validation positions to compare"
    )
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    args = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("run with python -I: training data is untrusted input")
    report = export(
        args.checkpoint, args.output, args.weights == "swa", args.positions, args.precision
    )
    print(f"Exported {args.output}: " + ", ".join(f"{k}={v:.2e}" for k, v in report.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
