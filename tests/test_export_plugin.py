# SPDX-License-Identifier: GPL-3.0-or-later
"""TensorRT expert-plugin export (markers: torch; tensorrt for the build and comparison)."""

from dataclasses import replace
import os
from pathlib import Path

import pytest

from test_model import TINY, _planes

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch", reason="PyTorch is installed by scripts/setup.ps1")
onnx = pytest.importorskip("onnx")

from mouse_moe.config import load_config  # noqa: E402
from mouse_moe.export_plugin import PLUGIN, export_plugin_model  # noqa: E402
from mouse_moe.model import MouseMoe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# mouse_moe_trt.dll, built by the Mouse engine's windows-moe CMake preset.
PLUGIN_LIBRARY = Path(os.environ.get("MOUSE_MOE_PLUGIN", ""))


def test_plugin_export_refuses_widths_the_plugin_cannot_vectorise(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin.dll"
    plugin.write_bytes(b"")
    odd = MouseMoe(replace(TINY, moe=replace(TINY.moe, expert_hidden=36)))
    with pytest.raises(ValueError, match="divisible by 8"):
        export_plugin_model(odd, _planes(4), tmp_path / "net.onnx", "ab" * 32, plugin)
    assert not (tmp_path / "net.onnx").exists()


def test_plugin_export_refuses_position_routing(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin.dll"
    plugin.write_bytes(b"")
    model = MouseMoe(replace(TINY, moe=replace(TINY.moe, routing="position")))
    with pytest.raises(ValueError, match="token routing"):
        export_plugin_model(model, _planes(4), tmp_path / "net.onnx", "ab" * 32, plugin)


def test_plugin_export_refuses_routing_beyond_the_plugin_limits(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin.dll"
    plugin.write_bytes(b"")
    wide = MouseMoe(replace(TINY, moe=replace(TINY.moe, experts=8, top_k=5)))
    with pytest.raises(ValueError, match="top-4 routing"):
        export_plugin_model(wide, _planes(4), tmp_path / "net.onnx", "ab" * 32, plugin)
    assert not (tmp_path / "net.onnx").exists()


@pytest.mark.tensorrt
# m1 again at top-4: the plugin's routing limit, not used by a shipped config yet.
@pytest.mark.parametrize(
    ("config", "top_k"), [("m1/stage1.toml", None), ("m20/m20.toml", None), ("m1/stage1.toml", 4)]
)
def test_plugin_export_matches_pytorch_on_tensorrt(
    tmp_path: Path, config: str, top_k: int | None
) -> None:
    if not PLUGIN_LIBRARY.is_file() or not torch.cuda.is_available():
        pytest.skip("set MOUSE_MOE_PLUGIN to the built plugin and run on a CUDA GPU")
    torch.manual_seed(5)
    shape = load_config(ROOT / "configs" / config).model
    if top_k is not None:
        shape = replace(shape, moe=replace(shape.moe, top_k=top_k))
    model = MouseMoe(shape)
    planes = (torch.rand(64, 112, 8, 8) > 0.85).float()
    output = tmp_path / "net.onnx"
    report = export_plugin_model(model, planes, output, "cd" * 32, PLUGIN_LIBRARY)
    graph = onnx.load(str(output))
    metadata = {p.key: p.value for p in graph.metadata_props}
    assert metadata["mouse.tensorrt_plugin"] == "MouseMoeExperts/1"
    assert sum(n.op_type == PLUGIN for n in graph.graph.node) == len(model.moe_layers())
    assert "Softplus" not in {n.op_type for n in graph.graph.node}
    assert report["top1_agreement"] >= 0.97


def test_tensorrt_lib_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from mouse_moe.label_teacher import TENSORRT_LIB_ENV, tensorrt_lib

    monkeypatch.delenv(TENSORRT_LIB_ENV, raising=False)
    with pytest.raises(RuntimeError, match=TENSORRT_LIB_ENV):
        tensorrt_lib()
    monkeypatch.setenv(TENSORRT_LIB_ENV, str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="not a folder"):
        tensorrt_lib()
    monkeypatch.setenv(TENSORRT_LIB_ENV, str(tmp_path))
    assert tensorrt_lib() == tmp_path
