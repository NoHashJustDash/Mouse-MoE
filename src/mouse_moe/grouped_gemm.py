# SPDX-License-Identifier: GPL-3.0-or-later
# ruff: noqa: N803  (Triton convention: sizes and tl.constexpr block shapes are upper case)
# mypy: disable-error-code="import-untyped,misc,no-untyped-def"
# (Triton ships no type information, and @triton.jit kernels take untyped pointers.)
"""Grouped matmul for the MoE experts as Triton kernels (CUDA, bfloat16).

``grouped_mm(rows, weights, ends)`` computes ``rows[a:b] @ weights[g]`` for every group
``g`` whose rows are ``a:b`` (rows sorted by group, ``ends`` the cumulative counts).
Group boundaries are read on the GPU only, so a training step never waits for the host
and can be captured as a CUDA graph.

``torch._grouped_mm`` did the same job on the RTX 5070 (sm_120) with 32x32-tile
kernels and a host copy of the offsets on every call: about 17 ms of a 60 ms ab_m1
forward pass (2026-10-25), where the arithmetic needs about 2.3 ms.

The kernels are registered with ``torch.library.triton_op``, so ``torch.compile``
traces them, and with an autograd formula: ``dX = dY @ W[g]^T`` (the same kernel with
transposed weights) and ``dW[g] = X[a:b]^T @ dY[a:b]``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

_FORWARD_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)
    for bm, bn, bk, w, s in [
        (128, 128, 32, 8, 3),
        (128, 64, 32, 4, 4),
        (64, 128, 32, 4, 4),
        (128, 128, 64, 8, 3),
        (64, 64, 64, 4, 4),
    ]
]
_WEIGHT_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)
    for bm, bn, bk, w, s in [
        (64, 128, 64, 8, 3),
        (128, 64, 64, 8, 3),
        (64, 64, 64, 4, 4),
        (32, 128, 128, 8, 3),
    ]
]


@triton.autotune(configs=_FORWARD_CONFIGS, key=["N", "K"])
@triton.jit
def _grouped_rows_kernel(
    a_ptr, b_ptr, c_ptr, ends_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bg, stride_bk, stride_bn,
    stride_cm, stride_cn,
    GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):  # fmt: skip
    """C[a:b] = A[a:b] @ B[g]. Each program takes one BLOCK_M row tile of one group."""
    tile = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Find the group of this tile: groups are laid out as ceil(count / BLOCK_M) tiles each.
    group = -1
    row_start = 0
    row_end = 0
    tiles_before = 0
    previous_end = 0
    for g in tl.static_range(GROUPS):
        end = tl.load(ends_ptr + g)
        tiles = (end - previous_end + BLOCK_M - 1) // BLOCK_M
        inside = (tile >= tiles_before) & (tile < tiles_before + tiles)
        group = tl.where(inside, g, group)
        row_start = tl.where(inside, previous_end + (tile - tiles_before) * BLOCK_M, row_start)
        row_end = tl.where(inside, end, row_end)
        tiles_before += tiles
        previous_end = end
    # Tiles past the last group (the grid's spare tiles) compute nothing and store nothing.
    used = group >= 0
    group = tl.maximum(group, 0)
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    row_ok = (rows < row_end) & used
    col_ok = cols < N
    a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
    b_ptrs = b_ptr + group * stride_bg + ks[:, None] * stride_bk + cols[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_ok = (k + ks) < K
        a = tl.load(a_ptrs, mask=row_ok[:, None] & k_ok[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_ok[:, None] & col_ok[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=row_ok[:, None] & col_ok[None, :])


@triton.autotune(configs=_WEIGHT_CONFIGS, key=["N", "K"])
@triton.jit
def _grouped_weight_kernel(
    a_ptr, d_ptr, w_ptr, ends_ptr,
    N, K,
    stride_am, stride_ak,
    stride_dm, stride_dn,
    stride_wg, stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):  # fmt: skip
    """W[g] = A[a:b]^T @ D[a:b], one (K tile, N tile) of one group per program."""
    group = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_n = tl.program_id(2)
    start = tl.where(group > 0, tl.load(ends_ptr + group - 1, mask=group > 0, other=0), 0)
    end = tl.load(ends_ptr + group)
    ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_ok = ks < K
    col_ok = cols < N
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for m in range(start, end, BLOCK_M):
        rows = m + tl.arange(0, BLOCK_M)
        row_ok = rows < end
        a = tl.load(
            a_ptr + rows[None, :] * stride_am + ks[:, None] * stride_ak,
            mask=k_ok[:, None] & row_ok[None, :],
            other=0.0,
        )
        d = tl.load(
            d_ptr + rows[:, None] * stride_dm + cols[None, :] * stride_dn,
            mask=row_ok[:, None] & col_ok[None, :],
            other=0.0,
        )
        acc = tl.dot(a, d, acc)
    w_ptrs = w_ptr + group * stride_wg + ks[:, None] * stride_wk + cols[None, :] * stride_wn
    tl.store(w_ptrs, acc.to(w_ptr.dtype.element_ty), mask=k_ok[:, None] & col_ok[None, :])


def _check(rows: Tensor, weights: Tensor, ends: Tensor) -> None:
    if not (rows.is_cuda and rows.dtype == weights.dtype == torch.bfloat16):
        raise ValueError("grouped_mm needs CUDA bfloat16 rows and weights")
    if rows.shape[1] != weights.shape[1] or ends.shape != (weights.shape[0],):
        raise ValueError(f"grouped_mm shapes {rows.shape} x {weights.shape}, ends {ends.shape}")


@triton_op("mouse_moe::grouped_mm", mutates_args={})
def grouped_mm(rows: Tensor, weights: Tensor, ends: Tensor) -> Tensor:
    """rows [M, K] sorted by group, weights [G, K, N], ends [G] int32 -> [M, N]."""
    _check(rows, weights, ends)
    m, k = rows.shape
    groups, _, n = weights.shape
    out = rows.new_empty(m, n)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        # At most one partial tile per group on top of the full ones.
        return (triton.cdiv(m, meta["BLOCK_M"]) + groups, triton.cdiv(n, meta["BLOCK_N"]))

    wrap_triton(_grouped_rows_kernel)[grid](
        rows, weights, out, ends,
        m, n, k,
        rows.stride(0), rows.stride(1),
        weights.stride(0), weights.stride(1), weights.stride(2),
        out.stride(0), out.stride(1),
        GROUPS=groups,
    )  # fmt: skip
    return out


@triton_op("mouse_moe::grouped_weight_grad", mutates_args={})
def grouped_weight_grad(rows: Tensor, grad: Tensor, ends: Tensor) -> Tensor:
    """rows [M, K], grad [M, N] sorted by group, ends [G] -> [G, K, N] per-group rows^T @ grad."""
    m, k = rows.shape
    n = grad.shape[1]
    groups = ends.shape[0]
    out = rows.new_empty(groups, k, n)

    def grid(meta: dict[str, int]) -> tuple[int, int, int]:
        return (groups, triton.cdiv(k, meta["BLOCK_K"]), triton.cdiv(n, meta["BLOCK_N"]))

    wrap_triton(_grouped_weight_kernel)[grid](
        rows, grad, out, ends,
        n, k,
        rows.stride(0), rows.stride(1),
        grad.stride(0), grad.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
    )  # fmt: skip
    return out


def _setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: Tensor) -> None:  # type: ignore[type-arg]
    rows, weights, ends = inputs
    ctx.save_for_backward(rows, weights, ends)


def _backward(
    ctx: torch.autograd.function.FunctionCtx, grad: Tensor
) -> tuple[Tensor, Tensor, None]:
    rows, weights, ends = ctx.saved_tensors  # type: ignore[attr-defined]
    grad = grad.contiguous()
    grad_rows = grouped_mm(grad, weights.transpose(1, 2), ends)
    grad_weights = grouped_weight_grad(rows, grad, ends)
    return grad_rows, grad_weights, None


grouped_mm.register_autograd(_backward, setup_context=_setup_context)
