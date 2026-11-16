# SPDX-License-Identifier: GPL-3.0-or-later
"""Export a Mouse-MoE checkpoint for TensorRT with the fused expert plugin, and verify it.

    python -I -m mouse_moe.export_plugin --checkpoint runs/<name>/checkpoints/<step>.pt \
        --output nets/<name>-trt.onnx --plugin-library <path to mouse_moe_trt.dll>

Each token-routed layer's experts become one ``trt.plugins::MouseMoeExperts`` node
(``engine/nn/trt_moe/moe_plugin.cu``); everything else matches ``export_onnx``. ONNX
Runtime's CPU provider cannot run that node, so the file is built with TensorRT in FP16
with the plugin loaded from ``--plugin-library`` and compared with PyTorch on a real
validation batch. Any failure deletes the output. The engine runs such a file only on its
TensorRT backend, from a build that includes the plugin (CMake preset windows-moe).
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

import onnx
import torch

from mouse_moe.export_onnx import (
    FP16_MAX_MEAN_KL,
    FP16_MAX_WDL_ERROR,
    FP16_MIN_TOP1,
    IR_VERSION,
    OPSET,
    ROUTING_TIE_MARGIN,
    _sha256,
    load_model,
    model_metadata,
    routing_margins,
    stable_softplus,
    validation_planes,
)
from mouse_moe.label_teacher import tensorrt_lib
from mouse_moe.model import EngineWrapper, MouseMoe, RoutedMoe

PLUGIN = "MouseMoeExperts"
PLUGIN_VERSION = "1"
# The plugin's CUTLASS GEMMs take any width and expert hidden width divisible by 8
# (8-half vector access); routing gives one lane of a half warp to each expert.
PLUGIN_ALIGNMENT = 8
MAX_EXPERTS = 16
MAX_TOP_K = 4
# The FP16 bounds of export_onnx. m1 on 35,530 test80 positions, legal moves only: top-1
# 99.4%, mean KL 4.9e-5, max WDL error 0.06.
MIN_TOP1 = FP16_MIN_TOP1
MAX_MEAN_KL = FP16_MAX_MEAN_KL
MAX_WDL_ERROR = FP16_MAX_WDL_ERROR


def _routed_layers(model: MouseMoe) -> list[RoutedMoe]:
    layers = [m for m in model.moe_layers() if isinstance(m, RoutedMoe)]
    if len(layers) != len(model.moe_layers()):
        raise ValueError("the TensorRT plugin covers token-routed MoE layers only (no soft MoE)")
    for layer in layers:
        width, hidden = layer.experts.w1.shape[1], layer.experts.w1.shape[2]
        if layer.position:
            raise ValueError("the TensorRT plugin supports token routing, not position routing")
        if width % PLUGIN_ALIGNMENT or hidden % PLUGIN_ALIGNMENT:
            raise ValueError(
                f"the TensorRT plugin needs widths divisible by {PLUGIN_ALIGNMENT}: "
                f"width {width}, hidden {hidden}"
            )
        if layer.cfg.experts > MAX_EXPERTS or layer.cfg.top_k > MAX_TOP_K:
            raise ValueError(
                f"the TensorRT plugin supports at most {MAX_EXPERTS} experts "
                f"and top-{MAX_TOP_K} routing"
            )
    return layers


class _TensorRt:
    """A one-profile FP16 TensorRT engine for an ONNX file using the expert plugin."""

    def __init__(self, path: Path, plugin_library: Path, batch: int) -> None:
        lib = tensorrt_lib()
        os.environ["PATH"] = str(lib) + os.pathsep + os.environ["PATH"]
        os.add_dll_directory(str(lib))
        import tensorrt as trt  # type: ignore[import-untyped]  # the wheel ships no stubs

        ctypes.CDLL(str(plugin_library))  # registers the plugin creator with TensorRT
        self.trt = trt
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(path)):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"TensorRT could not parse {path}: {errors}")
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        shape = (batch, 112, 8, 8)
        profile.set_shape(network.get_input(0).name, shape, shape, shape)
        config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
        self.context = self.engine.create_execution_context()

    def __call__(self, planes: torch.Tensor) -> dict[str, torch.Tensor]:
        trt = self.trt
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        planes = planes.cuda().contiguous()
        out: dict[str, torch.Tensor] = {}
        for name in names:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, tuple(planes.shape))
                self.context.set_tensor_address(name, planes.data_ptr())
        for name in names:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                out[name] = torch.empty(tuple(self.context.get_tensor_shape(name)), device="cuda")
                self.context.set_tensor_address(name, out[name].data_ptr())
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT inference failed")
        stream.synchronize()
        return {name: value.cpu() for name, value in out.items()}


def export_plugin_model(
    model: MouseMoe,
    reference_planes: torch.Tensor,
    output: Path,
    source_sha256: str,
    plugin_library: Path,
) -> dict[str, float]:
    """Export ``model`` with the expert plugin and verify it on TensorRT against PyTorch."""
    if not plugin_library.is_file():
        raise FileNotFoundError(f"plugin library not found: {plugin_library}")
    model.eval()
    layers = _routed_layers(model)
    with torch.no_grad():
        expected = EngineWrapper(model)(reference_planes)
    decided = routing_margins(model, reference_planes) >= ROUTING_TIE_MARGIN
    temporary = output.with_suffix(".tmp.onnx")
    try:
        for layer in layers:
            layer.plugin_export = True
        with torch.no_grad():
            torch.onnx.export(
                EngineWrapper(model),
                (reference_planes[:2],),
                str(temporary),
                input_names=["planes"],
                output_names=["policy", "wdl", "mlh"],
                dynamic_axes={name: {0: "batch"} for name in ("planes", "policy", "wdl", "mlh")},
                opset_version=OPSET,
                dynamo=False,
            )
    finally:
        for layer in layers:
            layer.plugin_export = False
    try:
        graph = onnx.load(str(temporary))
        graph.ir_version = IR_VERSION
        domains = {(item.domain, item.version) for item in graph.opset_import}
        if domains != {("", OPSET), ("trt.plugins", 1)}:
            raise AssertionError(f"plugin export used unexpected opsets: {domains}")
        nodes = [n for n in graph.graph.node if n.op_type == PLUGIN]
        if len(nodes) != len(layers) or any(n.domain != "trt.plugins" for n in nodes):
            raise AssertionError(f"expected {len(layers)} {PLUGIN} nodes, found {len(nodes)}")
        stable_softplus(graph.graph)
        metadata = model_metadata(model, source_sha256)
        metadata["mouse.tensorrt_plugin"] = f"{PLUGIN}/{PLUGIN_VERSION}"
        onnx.helper.set_model_props(graph, metadata)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(temporary))

        actual = _TensorRt(temporary, plugin_library, reference_planes.shape[0])(reference_planes)
        if not all(torch.isfinite(value).all() for value in actual.values()):
            raise AssertionError("TensorRT plugin export produced non-finite outputs")
        ref_policy = expected[0].double()[decided].log_softmax(1)
        trt_policy = actual["policy"].double()[decided].log_softmax(1)
        top1 = float((ref_policy.argmax(1) == trt_policy.argmax(1)).double().mean())
        mean_kl = float((ref_policy.exp() * (ref_policy - trt_policy)).sum(1).mean())
        wdl_error = float((actual["wdl"][decided] - expected[1][decided]).abs().max())
        report = {
            "routing_tied_boards": float((~decided).sum()),
            "top1_agreement": top1,
            "mean_policy_kl": mean_kl,
            "max_wdl_error": wdl_error,
        }
        if top1 < MIN_TOP1 or mean_kl > MAX_MEAN_KL or wdl_error > MAX_WDL_ERROR:
            raise AssertionError(f"TensorRT plugin export differs from PyTorch: {report}")
        os.replace(temporary, output)
        return report
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a Mouse-MoE checkpoint for TensorRT")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plugin-library", type=Path, required=True)
    parser.add_argument("--weights", choices=["swa", "raw"], default="swa")
    parser.add_argument(
        "--positions", type=int, default=256, help="validation positions to compare"
    )
    args = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("run with python -I: training data is untrusted input")
    model, config = load_model(args.checkpoint, args.weights == "swa")
    planes = torch.from_numpy(validation_planes(config, args.positions))
    report = export_plugin_model(
        model, planes, args.output, _sha256(args.checkpoint), args.plugin_library
    )
    print(f"Exported {args.output}: " + ", ".join(f"{k}={v:.3g}" for k, v in report.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
