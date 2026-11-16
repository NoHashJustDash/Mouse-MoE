# SPDX-License-Identifier: GPL-3.0-or-later
"""Triton grouped matmul for the MoE experts against a float32 loop (marker: torch, CUDA)."""

import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch", reason="PyTorch is installed by scripts/setup.ps1")
if not torch.cuda.is_available():
    pytest.skip("the Triton kernels need CUDA", allow_module_level=True)

from mouse_moe.grouped_gemm import grouped_mm  # noqa: E402
from mouse_moe.model import RoutedMoe  # noqa: E402
from test_model import TINY  # noqa: E402

GROUP_SIZES = {
    "balanced": [512] * 16,
    "one group takes everything": [0, 4096] + [0] * 14,
    # Partial tiles, single rows and empty groups between full ones.
    "odd sizes": [1, 2, 127, 129, 255, 0, 3000, 1, 64, 63, 65, 128, 0, 0, 7, 999],
}


def _reference(rows: "torch.Tensor", weights: "torch.Tensor", sizes: list[int]) -> "torch.Tensor":
    out, start = [], 0
    for group, size in enumerate(sizes):
        out.append(rows[start : start + size] @ weights[group])
        start += size
    return torch.cat(out)


@pytest.mark.parametrize("shape", [(256, 128), (128, 256), (64, 32)])
@pytest.mark.parametrize("name", list(GROUP_SIZES))
def test_grouped_mm_matches_a_float32_loop_in_value_and_gradients(
    name: str, shape: tuple[int, int]
) -> None:
    torch.manual_seed(0)
    sizes = GROUP_SIZES[name]
    k, n = shape
    ends = torch.cumsum(torch.tensor(sizes, device="cuda"), 0).to(torch.int32)
    rows = torch.randn(sum(sizes), k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(16, k, n, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(sum(sizes), n, device="cuda", dtype=torch.bfloat16)
    grouped_mm(rows, weights, ends).backward(grad)
    rows32 = rows.detach().float().requires_grad_()
    weights32 = weights.detach().float().requires_grad_()
    expected = _reference(rows32, weights32, sizes)
    expected.backward(grad.float())

    def relative(actual: "torch.Tensor", want: "torch.Tensor") -> float:
        return float((actual.float() - want).norm() / want.norm())

    # bfloat16 outputs: the measured error is the format's rounding, 1.7e-3.
    assert relative(grouped_mm(rows, weights, ends), expected) < 4e-3
    assert rows.grad is not None and weights.grad is not None
    assert relative(rows.grad, rows32.grad) < 4e-3
    assert relative(weights.grad, weights32.grad) < 4e-3


def test_grouped_mm_refuses_float32() -> None:
    rows = torch.randn(8, 64, device="cuda")
    with pytest.raises(ValueError, match="bfloat16"):
        grouped_mm(rows, torch.randn(2, 64, 32, device="cuda"), torch.tensor([4, 8], device="cuda"))


def test_cuda_training_layer_traces_whole_with_the_triton_kernels() -> None:
    torch.manual_seed(2)
    layer = RoutedMoe(TINY.d_model, TINY.moe, 1.0, position=False).cuda().train()
    traced = torch.compile(layer, fullgraph=True, backend="eager")
    x = torch.randn(4, 64, TINY.d_model, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        traced(x).float().sum().backward()
    assert layer.experts.w1.grad is not None and layer.experts.w1.grad.abs().sum() > 0
    torch._dynamo.reset()
