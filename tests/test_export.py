# SPDX-License-Identifier: GPL-3.0-or-later
"""ONNX export meets the Mouse engine contract (marker: torch)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from test_model import TINY, _planes

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch", reason="PyTorch is installed by scripts/setup.ps1")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from mouse_moe.export_onnx import dispatch_agrees, export_model  # noqa: E402
from mouse_moe.model import EngineWrapper, MouseMoe, RoutedMoe  # noqa: E402

REQUIRED_METADATA = {
    "mouse.schema_version": "1",
    "mouse.input_format": "1",
    "mouse.policy_format": "3",
    "mouse.value_format": "2",
    "mouse.moves_left_format": "1",
    "mouse.policy_mapping": "lc0-1858-v1",
    "mouse.precision": "fp32",
}


@pytest.mark.parametrize("routing", ["token", "position", "soft"])
def test_export_contract_and_dynamic_batch(tmp_path: Path, routing: str) -> None:
    torch.manual_seed(1)
    model = MouseMoe(replace(TINY, moe=replace(TINY.moe, routing=routing)))
    output = tmp_path / "net.onnx"
    report = export_model(model, _planes(6), output, "ab" * 32)
    assert all(value < 1e-3 for value in report.values())

    graph = onnx.load(str(output))
    assert graph.ir_version == 8
    assert {(o.domain, o.version) for o in graph.opset_import} == {("", 17)}
    metadata = {p.key: p.value for p in graph.metadata_props}
    for key, value in REQUIRED_METADATA.items():
        assert metadata[key] == value
    assert int(metadata["mouse.learned_parameter_count"]) > 0
    assert [i.name for i in graph.graph.input] == ["planes"]
    assert [o.name for o in graph.graph.output] == ["policy", "wdl", "mlh"]
    for value in (*graph.graph.input, *graph.graph.output):
        assert value.type.tensor_type.shape.dim[0].dim_param == "batch"

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    for count in (1, 5, 64):  # batch sizes different from the traced one
        planes = _planes(count) + 0.05 * torch.randn(count, 112, 8, 8)
        policy, wdl, mlh = session.run(None, {"planes": planes.numpy()})
        assert policy.shape == (count, 1858) and wdl.shape == (count, 3)
        np.testing.assert_allclose(wdl.sum(axis=1), 1.0, atol=1e-5)
        assert (mlh >= 0).all()


@pytest.mark.parametrize("routing", ["token", "position"])
def test_routed_export_has_static_shapes_and_no_dangling_outputs(
    tmp_path: Path, routing: str
) -> None:
    # Regressions: NonZero dispatch broke TensorRT CUDA-graph capture; an unused TopK
    # output became an extra model output in ORT's TensorRT context dump; and TensorRT
    # 10.9/10.16.1 returned TopK/ArgMax indices in [square, board] order inside the
    # full graph (ab_m1 top-1 agreement 1%), so the export has no index-producing op.
    torch.manual_seed(2)
    model = MouseMoe(replace(TINY, moe=replace(TINY.moe, routing=routing)))
    output = tmp_path / "net.onnx"
    export_model(model, _planes(4), output, "cd" * 32)
    graph = onnx.load(str(output)).graph
    ops = {node.op_type for node in graph.node}
    assert not ops & {"NonZero", "TopK", "ArgMax", "ArgMin"}
    consumed = {name for node in graph.node for name in node.input}
    consumed |= {o.name for o in graph.output}
    for node in graph.node:
        if len(node.output) > 1:
            assert all(name in consumed for name in node.output), node.name


def test_export_has_no_softplus_that_overflows_in_tensorrt_fp16(tmp_path: Path) -> None:
    # Regression: TensorRT 10.9 FP16 computes Softplus as log(1 + exp(x)). Moves-left
    # pre-activations reach hundreds of plies, exp overflows FP16, and m1 returned an
    # infinite mlh on 982 of 35,530 real test80 positions (2026-10-27). Mish uses Softplus
    # too. The export writes it as relu(x) + log(1 + exp(-|x|)), the same function.
    torch.manual_seed(4)
    model = MouseMoe(TINY)
    output = tmp_path / "net.onnx"
    export_model(model, _planes(4), output, "12" * 32)
    assert "Softplus" not in {node.op_type for node in onnx.load(str(output)).graph.node}


def test_stable_softplus_rewrite_matches_softplus_at_large_values() -> None:
    from onnx import TensorProto, helper

    from mouse_moe.export_onnx import stable_softplus

    graph = helper.make_graph(
        [helper.make_node("Softplus", ["x"], ["y"], name="s")],
        "softplus",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [7])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [7])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    assert stable_softplus(model.graph) == 1
    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    x = np.array([-300.0, -20.0, -1.0, 0.0, 1.0, 20.0, 300.0], dtype=np.float32)
    (y,) = session.run(None, {"x": x})
    np.testing.assert_allclose(
        y, torch.nn.functional.softplus(torch.from_numpy(x)).numpy(), rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("routing", ["token", "position"])
def test_export_runs_one_board_repeated_like_a_search_batch(tmp_path: Path, routing: str) -> None:
    # A search batch is near-identical positions, so every expert's load is maximally
    # skewed; the tiled export must still match PyTorch exactly.
    torch.manual_seed(3)
    model = MouseMoe(replace(TINY, moe=replace(TINY.moe, routing=routing))).eval()
    output = tmp_path / "net.onnx"
    export_model(model, _planes(4), output, "ef" * 32)
    planes = (_planes(1) + 0.05 * torch.randn(1, 112, 8, 8)).repeat(40, 1, 1, 1)
    with torch.no_grad():
        expected = EngineWrapper(model)(planes)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"planes": planes.numpy()})
    for want, got in zip(expected, actual, strict=True):
        np.testing.assert_allclose(got, want.numpy(), atol=1e-4, rtol=1e-4)


def test_dispatch_check_allows_relative_rounding_on_large_values() -> None:
    # Regression: an absolute-only 1e-5 bound rejected the real ab_m2 export over
    # summation-order rounding on moves-left (230 plies, 6.9e-5 apart; 3e-7 relative).
    moves_left = torch.tensor([230.0, 161.0, 3.0])
    assert dispatch_agrees(moves_left, moves_left + torch.tensor([6.9e-5, -4e-5, 1e-6]))
    assert not dispatch_agrees(moves_left, moves_left + torch.tensor([0.0, 0.01, 0.0]))
    assert not dispatch_agrees(torch.tensor([0.5]), torch.tensor([0.5001]))


def test_routing_margin_is_the_gap_below_the_last_chosen_expert() -> None:
    layer = RoutedMoe(TINY.d_model, TINY.moe, 1.0, position=False).eval()
    torch.nn.init.zeros_(layer.router.weight)  # every expert scores 1/4
    x = torch.randn(3, 5, TINY.d_model)
    layer.balance_bias.copy_(torch.tensor([0.3, 0.2, 0.1, 0.1]))
    torch.testing.assert_close(layer.routing_margin(x), torch.full((3,), 0.1))
    layer.balance_bias.copy_(torch.tensor([0.3, 0.1, 0.1, 0.0]))  # 2nd and 3rd tie
    assert layer.routing_margin(x).tolist() == [0.0, 0.0, 0.0]


def test_export_refuses_when_rounding_decides_too_much_routing(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = MouseMoe(TINY).eval()
    for layer in model.moe_layers():
        assert isinstance(layer, RoutedMoe)
        torch.nn.init.zeros_(layer.router.weight)  # all experts tie on every square
    with pytest.raises(AssertionError, match="routing ties"):
        export_model(model, _planes(4), tmp_path / "net.onnx", "0" * 64)
    assert not (tmp_path / "net.onnx").exists()


def test_fp16_export_keeps_float32_io_and_routers(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = MouseMoe(TINY)
    output = tmp_path / "net.onnx"
    report = export_model(model, _planes(6), output, "ab" * 32, precision="fp16")
    assert report["top1_agreement"] >= 0.97 and report["mean_policy_kl"] <= 5e-4

    graph = onnx.load(str(output))
    float32 = onnx.TensorProto.FLOAT
    assert [v.type.tensor_type.elem_type for v in graph.graph.input] == [float32]
    assert {v.type.tensor_type.elem_type for v in graph.graph.output} == {float32}
    assert {p.key: p.value for p in graph.metadata_props}["mouse.precision"] == "fp16"
    assert any(t.data_type == onnx.TensorProto.FLOAT16 for t in graph.graph.initializer)
    # The only float32 weight matrices are the routers, one [width, experts] per MoE layer.
    matrices = [
        list(t.dims) for t in graph.graph.initializer if t.data_type == float32 and len(t.dims) == 2
    ]
    assert matrices == [[TINY.d_model, TINY.moe.experts]] * len(model.moe_layers())
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    policy, wdl, mlh = session.run(None, {"planes": _planes(3).numpy()})
    assert policy.dtype == np.float32 and np.isfinite(policy).all()
    np.testing.assert_allclose(wdl.sum(axis=1), 1.0, atol=1e-3)
    assert (mlh >= 0).all()


def test_export_refuses_an_unknown_precision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="precision"):
        export_model(MouseMoe(TINY), _planes(2), tmp_path / "net.onnx", "ab" * 32, "bf16")
