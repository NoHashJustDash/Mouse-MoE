# SPDX-License-Identifier: GPL-3.0-or-later
"""Mouse-MoE: Chessformer-style encoder with Mixture-of-Experts feed-forward layers.

Design: ``docs/ARCHITECTURE.md``. Input/output contract
matches the Mouse engine (``engine/nn/model_metadata.cpp``): ``planes`` float32
[B, 112, 8, 8] with the raw rule-50 count in plane 109; outputs ``policy``
logits [B, 1858], ``wdl`` probabilities [B, 3], ``mlh`` [B, 1] (non-negative).
"""

from __future__ import annotations

import itertools
import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mouse_moe.config import ModelConfig, MoeConfig
from mouse_moe.policy_map import POLICY_SIZE, PROMOTION_CHANNELS, attention_gather_tables
from mouse_moe.v6 import AUX_FEATURES, INPUT_PLANES, PIECE_PLANES

TOKENS = 64
RULE50_PLANE = 109
RULE50_SCALE = 1.0 / 99.0  # lczero-training's classical-format divisor
# Export dispatch pads each expert's assignments to tiles of this many rows. ab_m1 in
# TensorRT 10.9 FP32, batch 96 (2026-10-21): 64 rows 6.8 ms, 128 rows 5.2 ms, 256 rows 5.9 ms.
EXPORT_TILE_ROWS = 128


def expand_planes(packed: Tensor, aux: Tensor) -> Tensor:
    """GPU twin of ``v6.expand_planes``: uint8 [N, 832] + float [N, 8] -> [N, 112, 8, 8]."""
    count = packed.shape[0]
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    pieces = bits.reshape(count, PIECE_PLANES, TOKENS).to(aux.dtype)
    extra = aux.reshape(count, AUX_FEATURES, 1).expand(count, AUX_FEATURES, TOKENS)
    return torch.cat([pieces, extra], dim=1).reshape(count, INPUT_PLANES, 8, 8)


def top_k_ids_without_index_ops(scores: Tensor, k: int) -> Tensor:
    """The k highest-scoring columns per row, in ascending column order (int64 [N, k]).

    Used on the export path instead of TopK. TensorRT 10.9 and 10.16.1 returned TopK
    and ArgMax *indices* in [square, board] order inside the full Mouse-MoE graph while
    the values stayed [board, square] (ab_m1: TensorRT top-1 agreement 1%). This uses
    only ReduceMax, comparisons, CumSum and sums, so every intermediate is a value
    tensor. On an exact tie for the k-th place the lowest columns win.
    """
    columns = scores.shape[-1]
    kth = scores.amax(dim=-1, keepdim=True)
    for _ in range(k - 1):
        # Step down to the next distinct value only while fewer than k columns reach
        # the threshold, so a value tied across several columns keeps all of them.
        lower = scores.masked_fill(scores >= kth, float("-inf")).amax(dim=-1, keepdim=True)
        enough = (scores >= kth).sum(dim=-1, keepdim=True) >= k
        kth = torch.where(enough, kth, lower)
    # Fewer than k columns lie strictly above kth; ties at kth fill the remaining places.
    above = (scores > kth).to(torch.int32)
    tied = (scores == kth).to(torch.int32)
    places = k - above.sum(dim=-1, keepdim=True)
    candidates = above + tied * (torch.cumsum(tied, dim=-1) <= places).to(torch.int32)
    order = torch.cumsum(candidates, dim=-1) * candidates  # 1..k on the chosen columns
    ids = torch.arange(columns, device=scores.device, dtype=torch.int32)
    chosen = [((order == i + 1).to(torch.int32) * ids).sum(dim=-1, keepdim=True) for i in range(k)]
    return torch.cat(chosen, dim=-1).to(torch.int64)


def _deepnorm(layers: int) -> tuple[float, float]:
    """DeepNet residual scale alpha and init gain beta for an encoder (Wang et al. 2022)."""
    return (2.0 * layers) ** 0.25, (8.0 * layers) ** -0.25


class DenseFfn(nn.Module):
    def __init__(self, d_model: int, hidden: int, gain: float) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, hidden)
        self.down = nn.Linear(hidden, d_model)
        for layer in (self.up, self.down):
            nn.init.xavier_normal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.mish(self.up(x)))


def expert_counts(selected: Tensor, experts: int) -> Tensor:
    """Assignments per expert (int32 [experts]); unlike ``bincount`` its size is static."""
    ids = selected.reshape(-1)
    counts = torch.zeros(experts, dtype=torch.int32, device=ids.device)
    return counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))


def grouped_matmul(rows: Tensor, weights: Tensor, ends: Tensor) -> Tensor:
    """rows [N, in] sorted by group, weights [G, in, out], ends [G] cumulative row counts.

    CUDA bfloat16 runs the Triton kernels in ``mouse_moe.grouped_gemm`` (no host sync).
    CPU bfloat16 uses ``torch._grouped_mm``, which takes bfloat16 only (PyTorch 2.9).
    Float32 (the fp32 precision setting and the CPU tests) loops over the groups,
    which reads ``ends`` back to the host.
    """
    if rows.dtype == torch.bfloat16 and rows.is_cuda:
        from mouse_moe.grouped_gemm import grouped_mm

        return cast(Tensor, grouped_mm(rows, weights, ends))
    if rows.dtype == torch.bfloat16:
        return torch._grouped_mm(rows, weights, offs=ends)
    bounds = [0, *ends.tolist()]
    parts = [rows[a:b] @ weights[g] for g, (a, b) in enumerate(itertools.pairwise(bounds))]
    return torch.cat(parts)


class _GatherRows(torch.autograd.Function):
    """``source[index]`` whose backward is a gather too, not ``index_add_``.

    Row ``i`` of the result is ``source[index[i]]``. Source row ``r`` was gathered into
    the rows ``spread[r * fanout : (r + 1) * fanout]``, so its gradient is their sum.
    bfloat16 ``index_add_`` (the autograd default) took 11 of 115 ms of GPU time per
    ab_m1 step on the RTX 5070 (2026-10-25).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        source: Tensor,
        index: Tensor,
        spread: Tensor,
        fanout: int,
    ) -> Tensor:
        ctx.save_for_backward(spread)
        ctx.fanout = fanout  # type: ignore[attr-defined]
        return source.index_select(0, index)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx, grad: Tensor
    ) -> tuple[Tensor, None, None, None]:
        (spread,) = ctx.saved_tensors  # type: ignore[attr-defined]
        fanout: int = ctx.fanout  # type: ignore[attr-defined]
        rows = grad.index_select(0, spread)
        return rows.reshape(-1, fanout, grad.shape[-1]).sum(dim=1), None, None, None


def gather_rows(source: Tensor, index: Tensor, spread: Tensor, fanout: int) -> Tensor:
    """``source[index]`` with a gather backward; see ``_GatherRows``."""
    return cast(Tensor, _GatherRows.apply(source, index, spread, fanout))  # type: ignore[no-untyped-call]


class Experts(nn.Module):
    """E independent two-layer Mish FFNs stored as stacked weights."""

    def __init__(self, experts: int, d_model: int, hidden: int, gain: float) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(experts, d_model, hidden))
        self.b1 = nn.Parameter(torch.zeros(experts, hidden))
        self.w2 = nn.Parameter(torch.empty(experts, hidden, d_model))
        self.b2 = nn.Parameter(torch.zeros(experts, d_model))
        for e in range(experts):
            nn.init.xavier_normal_(self.w1.data[e].T, gain=gain)
            nn.init.xavier_normal_(self.w2.data[e].T, gain=gain)


class _TensorRtExperts(torch.autograd.Function):
    """ONNX export only: the routed experts as one trt.plugins::MouseMoeExperts node.

    The plugin (engine/nn/trt_moe/moe_plugin.cu) reproduces routing and the experts of
    the export path. Tracing needs only the output shape, so forward returns zeros; the
    exported file is verified on TensorRT (mouse_moe.export_plugin).
    """

    @staticmethod
    def forward(
        ctx: object,
        flat: Tensor,
        router: Tensor,
        bias: Tensor,
        w1: Tensor,
        b1: Tensor,
        w2: Tensor,
        b2: Tensor,
        top_k: int,
    ) -> Tensor:
        return torch.zeros_like(flat)

    @staticmethod
    def symbolic(  # type: ignore[no-untyped-def]
        g, flat, router, bias, w1, b1, w2, b2, top_k
    ):
        return g.op(
            "trt.plugins::MouseMoeExperts", flat, router, bias, w1, b1, w2, b2, top_k_i=top_k
        )


class RoutedMoe(nn.Module):
    """Shared expert + top-k routed experts with auxiliary-loss-free bias balancing.

    ``position`` routing makes one decision per board (mean-pooled tokens) and sends
    all 64 squares to the same experts; ``token`` routing decides per square.
    Selection uses ``scores + balance_bias``; gate weights use unbiased scores
    renormalised over the selected experts (Wang et al. 2024, arXiv 2408.15664).
    """

    def __init__(self, d_model: int, cfg: MoeConfig, gain: float, position: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.position = position
        self.router = nn.Linear(d_model, cfg.experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.experts = Experts(cfg.experts, d_model, cfg.expert_hidden, gain)
        self.shared = DenseFfn(d_model, cfg.shared_hidden, gain) if cfg.shared_hidden else None
        self.register_buffer("balance_bias", torch.zeros(cfg.experts))
        self.register_buffer("load", torch.zeros(cfg.experts), persistent=False)
        self.aux_loss: Tensor | None = None
        self.export_mode = False
        self.plugin_export = False  # ONNX export through the TensorRT expert plugin

    def _scores(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Router logits, softmax scores and the balance-biased scores experts are picked by."""
        batch, tokens, _ = x.shape
        router_input = x.mean(dim=1) if self.position else x.reshape(batch * tokens, -1)
        # FP16 exports keep the router in float32 (export_onnx --precision fp16), so which
        # experts are chosen does not depend on half-precision rounding.
        logits = self.router(router_input.to(self.router.weight.dtype)).float()
        scores = logits.softmax(dim=-1)
        return logits, scores, scores + self.balance_bias

    def routing_margin(self, x: Tensor) -> Tensor:
        """Per board, the smallest gap between the k-th and (k+1)-th biased score.

        When the gap is a few float32 rounding steps, which expert is chosen depends on
        how a backend rounds the router, not on the weights.
        """
        _, _, biased = self._scores(x)
        if self.cfg.experts <= self.cfg.top_k:
            return biased.new_full((x.shape[0],), float("inf"))
        top = biased.topk(self.cfg.top_k + 1, dim=-1).values
        return (top[:, -2] - top[:, -1]).reshape(x.shape[0], -1).amin(dim=1)

    def _route(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tokens = x.shape[1]
        logits, scores, biased = self._scores(x)
        if self.export_mode:
            selected = top_k_ids_without_index_ops(biased, self.cfg.top_k)
        else:
            selected = torch.topk(biased, self.cfg.top_k, dim=-1).indices
        gates = scores.gather(1, selected)
        if self.cfg.top_k > 1:
            # Renormalising over the chosen experts keeps the router differentiable
            # through the ratio between them.
            gates = gates / gates.sum(dim=-1, keepdim=True)
        # top-1 keeps the raw selected probability (Switch Transformer). Renormalising a
        # single gate gives p/p = 1, which leaves the router no task gradient at all.
        if self.position:
            selected = selected.repeat_interleave(tokens, dim=0)
            gates = gates.repeat_interleave(tokens, dim=0)
        return logits, scores, selected, gates

    def _losses(self, logits: Tensor, scores: Tensor, selected: Tensor) -> Tensor:
        experts = self.cfg.experts
        decisions = scores.shape[0]
        counts = expert_counts(selected[:: (TOKENS if self.position else 1)], experts).float()
        with torch.no_grad():
            self.load += counts
        fraction = counts / (decisions * self.cfg.top_k)
        balance = experts * (fraction * scores.mean(dim=0)).sum()
        z_loss = torch.logsumexp(logits, dim=-1).square().mean()
        return self.cfg.aux_loss_weight * balance + self.cfg.z_loss_weight * z_loss

    def _dispatch_grouped(self, flat: Tensor, selected: Tensor, gates: Tensor) -> Tensor:
        """Training path: assignments sorted by expert, one grouped matmul per weight.

        Shapes stay fixed and nothing is read back to the host, so the layer compiles
        whole (``torch.compile``). Expert biases are added through a one-hot matmul:
        indexing ``b1[expert_of_row]`` back-propagates as a 131k-into-16-row scatter
        that took 170 of 287 ms per ab_m1 step on the RTX 5070 (2026-10-25). Each
        token's k outputs are gathered back and summed in ``flat``'s precision.
        """
        e, experts, top_k = self.experts, self.cfg.experts, self.cfg.top_k
        device = flat.device.type
        autocast = torch.is_autocast_enabled(device)
        dtype = torch.get_autocast_dtype(device) if autocast else flat.dtype
        expert_ids = selected.reshape(-1)
        order = torch.argsort(expert_ids, stable=True)
        ends = torch.cumsum(expert_counts(expert_ids, experts), dim=0, dtype=torch.int32)
        # slot[a] is where assignment a = token * k + choice sits in expert order.
        slot = torch.empty_like(order).scatter_(
            0, order, torch.arange(order.numel(), device=order.device)
        )
        rows = gather_rows(flat.to(dtype), order // top_k, slot, top_k)
        onehot = F.one_hot(expert_ids[order], experts).to(dtype)
        hidden = F.mish(grouped_matmul(rows, e.w1.to(dtype), ends) + onehot @ e.b1.to(dtype))
        y = grouped_matmul(hidden, e.w2.to(dtype), ends) + onehot @ e.b2.to(dtype)
        picked = gather_rows(y, slot, order, 1).reshape(-1, top_k, y.shape[-1])
        return (picked.to(flat.dtype) * gates.to(flat.dtype)[:, :, None]).sum(dim=1)

    def _dispatch_export(self, flat: Tensor, selected: Tensor, gates: Tensor) -> Tensor:
        """Export path: sparse, drop-free, static-shaped dispatch over padded tiles.

        Each (token, choice) assignment gets a slot in a buffer where every expert's
        assignments start on an ``EXPORT_TILE_ROWS`` boundary, so each tile belongs to one
        expert and all experts run as one batched matmul with per-tile gathered weights.
        The tile count is a static bound, ceil(T * k / R) + E, so TensorRT builds one
        engine per shape and CUDA-graph capture works (``NonZero`` dispatch broke it).
        Nothing is dropped: one board can route all 64 squares to the same experts
        (ab_m1: max/mean expert load 8.0), so fixed per-expert capacity would lose tokens.

        Ranks are counted per token over earlier tokens, which is valid because a
        token's k choices are distinct experts. Padding slots read token 0 and their
        outputs are never gathered back.
        """
        e = self.experts
        experts, top_k, rows = self.cfg.experts, self.cfg.top_k, EXPORT_TILE_ROWS
        tokens, width = flat.shape
        per_token = torch.zeros(tokens, experts, dtype=torch.int32, device=flat.device)
        per_token = per_token.scatter(1, selected, 1)
        earlier = torch.cumsum(per_token, dim=0) - per_token
        rank = torch.gather(earlier, 1, selected)
        counts = per_token.sum(dim=0)
        padded = (counts + rows - 1) // rows * rows
        ends = torch.cumsum(padded, dim=0)
        slot = (ends - padded)[selected] + rank  # [T, k]
        tiles = (tokens * top_k + rows - 1) // rows + experts
        token_of = torch.arange(tokens, device=flat.device)[:, None].expand(tokens, top_k)
        slot_token = torch.zeros(tiles * rows, dtype=torch.int64, device=flat.device)
        slot_token = slot_token.scatter(0, slot.reshape(-1).to(torch.int64), token_of.reshape(-1))
        tile_start = torch.arange(tiles, device=flat.device, dtype=ends.dtype) * rows
        tile_expert = (tile_start[:, None] >= ends[None, :]).to(torch.int64).sum(dim=1)
        tile_expert = tile_expert.clamp(max=experts - 1)
        x = flat[slot_token].reshape(tiles, rows, width)
        hidden = F.mish(torch.bmm(x, e.w1[tile_expert]) + e.b1[tile_expert][:, None, :])
        y = torch.bmm(hidden, e.w2[tile_expert]) + e.b2[tile_expert][:, None, :]
        picked = y.reshape(tiles * rows, width)[slot.to(torch.int64)]  # [T, k, width]
        return (picked * gates.to(flat.dtype)[:, :, None]).sum(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, width = x.shape
        flat = x.reshape(batch * tokens, width)
        if self.plugin_export:
            e = self.experts
            out: Tensor = _TensorRtExperts.apply(  # type: ignore[no-untyped-call]
                flat, self.router.weight, self.balance_bias, e.w1, e.b1, e.w2, e.b2, self.cfg.top_k
            )
            if self.shared is not None:
                out = out + self.shared(flat)
            return out.reshape(batch, tokens, width)
        logits, scores, selected, gates = self._route(x)
        if self.training:
            self.aux_loss = self._losses(logits, scores, selected)
        dispatch = self._dispatch_export if self.export_mode else self._dispatch_grouped
        out = dispatch(flat, selected, gates)
        if self.shared is not None:
            out = out + self.shared(flat)
        return out.reshape(batch, tokens, width)

    @torch.no_grad()
    def update_balance(self) -> None:
        """Nudge each expert's selection bias toward equal load; call after optimizer.step().

        A step that routed nothing leaves the biases as they are. Written without a
        branch, so it never waits on the GPU.
        """
        step = self.cfg.bias_update_rate * torch.sign(self.load.mean() - self.load)
        self.balance_bias += torch.where(self.load.sum() > 0, step, 0.0)
        # A shift shared by every bias changes no selection; centring stops the biases
        # drifting up together (m20 ended its 500k steps with all 16 between +4 and +7).
        self.balance_bias -= self.balance_bias.mean()
        self.load.zero_()


class SoftMoe(nn.Module):
    """Soft MoE (Puigcerver et al., arXiv 2308.00951) plus the shared expert.

    Each board's 64 tokens are mixed into experts x slots soft slots, processed by
    their expert and mixed back. Static shapes, no routing discontinuity, no
    balancing loss.
    """

    def __init__(self, d_model: int, cfg: MoeConfig, gain: float) -> None:
        super().__init__()
        self.cfg = cfg
        slots = cfg.experts * cfg.slots_per_expert
        self.phi = nn.Parameter(torch.randn(d_model, slots) / math.sqrt(d_model))
        self.scale = nn.Parameter(torch.ones(()))
        self.experts = Experts(cfg.experts, d_model, cfg.expert_hidden, gain)
        self.shared = DenseFfn(d_model, cfg.shared_hidden, gain) if cfg.shared_hidden else None
        self.aux_loss: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, width = x.shape
        e, s = self.cfg.experts, self.cfg.slots_per_expert
        logits = (F.normalize(x, dim=-1) @ F.normalize(self.phi, dim=0)) * self.scale
        dispatch = logits.softmax(dim=1)  # over tokens
        combine = logits.softmax(dim=2)  # over slots
        slots = (dispatch.transpose(1, 2) @ x).reshape(batch, e, s, width)
        hidden = F.mish(
            torch.einsum("besd,edf->besf", slots, self.experts.w1)
            + self.experts.b1[None, :, None, :]
        )
        processed = (
            torch.einsum("besf,efd->besd", hidden, self.experts.w2)
            + self.experts.b2[None, :, None, :]
        ).reshape(batch, e * s, width)
        out = combine @ processed
        if self.shared is not None:
            out = out + self.shared(x)
        if self.training:
            self.aux_loss = x.new_zeros(())
        return out


class Gab(nn.Module):
    """Geometric Attention Bias (Chessformer, ICLR 2026; Lc0 'smolgen')."""

    def __init__(self, d_model: int, heads: int, compress: int, hidden: int, per_head: int) -> None:
        super().__init__()
        self.heads, self.per_head = heads, per_head
        self.compress = nn.Linear(d_model, compress, bias=False)
        self.summary = nn.Linear(TOKENS * compress, hidden)
        self.summary_norm = nn.LayerNorm(hidden, bias=False)
        self.generate = nn.Linear(hidden, heads * per_head)
        self.generate_norm = nn.LayerNorm(per_head, bias=False)

    def forward(self, x: Tensor, shared: nn.Linear) -> Tensor:
        batch = x.shape[0]
        summary = self.summary_norm(F.gelu(self.summary(self.compress(x).reshape(batch, -1))))
        per_head = self.generate(summary).reshape(batch, self.heads, self.per_head)
        bias = shared(self.generate_norm(F.gelu(per_head)))
        return bias.reshape(batch, self.heads, TOKENS, TOKENS)


class Attention(nn.Module):
    def __init__(self, d_model: int, heads: int, gain: float) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, d_model // heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        nn.init.xavier_normal_(self.q.weight)
        nn.init.xavier_normal_(self.k.weight)
        nn.init.xavier_normal_(self.v.weight, gain=gain)
        nn.init.xavier_normal_(self.out.weight, gain=gain)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, bias: Tensor) -> Tensor:
        batch, tokens, width = x.shape

        def split(t: Tensor) -> Tensor:
            return t.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        # Fused attention with the GAB bias as an additive mask: on CUDA this uses
        # the memory-efficient kernel (fp32 softmax accumulation, probabilities
        # recomputed in backward instead of stored); ONNX export decomposes it.
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.out(context.transpose(1, 2).reshape(batch, tokens, width))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, index: int) -> None:
        super().__init__()
        self.alpha, gain = _deepnorm(cfg.layers)
        self.gab = Gab(cfg.d_model, cfg.heads, cfg.gab.compress, cfg.gab.hidden, cfg.gab.per_head)
        self.attention = Attention(cfg.d_model, cfg.heads, gain)
        self.norm1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.norm2 = nn.LayerNorm(cfg.d_model, bias=False)
        moe_here = cfg.ffn == "moe" and index % cfg.moe.every == cfg.moe.every - 1
        self.ffn: nn.Module
        if not moe_here:
            self.ffn = DenseFfn(cfg.d_model, cfg.dense_hidden, gain)
        elif cfg.moe.routing == "soft":
            self.ffn = SoftMoe(cfg.d_model, cfg.moe, gain)
        else:
            self.ffn = RoutedMoe(cfg.d_model, cfg.moe, gain, position=cfg.moe.routing == "position")

    def forward(self, x: Tensor, gab_shared: nn.Linear) -> Tensor:
        x = self.norm1(self.alpha * x + self.attention(x, self.gab(x, gab_shared)))
        return self.norm2(self.alpha * x + self.ffn(x))


class PolicyHead(nn.Module):
    """Source->destination attention policy mapped onto Lc0's 1858 move order."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.promotion = nn.Linear(d_model, PROMOTION_CHANNELS)
        self.scale = d_model**-0.5
        base, promo = attention_gather_tables()
        self.register_buffer("base_index", torch.from_numpy(base), persistent=False)
        self.register_buffer("promo_index", torch.from_numpy(promo), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        query, key = self.query(x), self.key(x)
        moves = (query @ key.transpose(1, 2)).reshape(batch, TOKENS * TOKENS) * self.scale
        promo = self.promotion(key[:, 56:64, :]).reshape(batch, 8 * PROMOTION_CHANNELS)
        promo = torch.cat([promo, promo.new_zeros(batch, 1)], dim=1)
        return moves.index_select(1, self.base_index) + promo.index_select(1, self.promo_index)


class TokenHead(nn.Module):
    """Per-square projection, flatten, two dense layers (Lc0 value / moves-left style)."""

    def __init__(self, d_model: int, channels: int, hidden: int, outputs: int) -> None:
        super().__init__()
        self.project = nn.Linear(d_model, channels)
        self.hidden = nn.Linear(TOKENS * channels, hidden)
        self.output = nn.Linear(hidden, outputs)

    def forward(self, x: Tensor) -> Tensor:
        flat = F.mish(self.project(x)).reshape(x.shape[0], -1)
        return self.output(F.mish(self.hidden(flat)))


class MouseMoe(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        scale = torch.ones(INPUT_PLANES)
        scale[RULE50_PLANE] = RULE50_SCALE
        self.register_buffer("input_scale", scale, persistent=False)
        self.embed = nn.Linear(INPUT_PLANES, cfg.d_model)
        self.position = nn.Parameter(torch.zeros(TOKENS, cfg.d_model))
        self.embed_norm = nn.LayerNorm(cfg.d_model, bias=False)
        self.gab_shared = nn.Linear(cfg.gab.per_head, TOKENS * TOKENS, bias=False)
        nn.init.zeros_(self.gab_shared.weight)  # start as plain attention
        self.layers = nn.ModuleList(EncoderLayer(cfg, i) for i in range(cfg.layers))
        self.policy = PolicyHead(cfg.d_model)
        self.value = TokenHead(cfg.d_model, cfg.value_channels, cfg.value_hidden, 3)
        self.moves_left = TokenHead(cfg.d_model, cfg.mlh_channels, cfg.mlh_hidden, 1)

    def forward(self, planes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (policy logits [B,1858], WDL logits [B,3], moves-left [B,1] >= 0)."""
        batch = planes.shape[0]
        x = planes.reshape(batch, INPUT_PLANES, TOKENS).transpose(1, 2) * self.input_scale
        x = self.embed_norm(F.mish(self.embed(x) + self.position))
        for layer in self.layers:
            x = layer(x, self.gab_shared)
        # Softplus, not ReLU: the engine needs mlh >= 0, and a ReLU output unit can die
        # permanently (zero gradient), which test_tiny_model_overfits_one_batch caught.
        return self.policy(x), self.value(x), F.softplus(self.moves_left(x))

    def moe_layers(self) -> list[RoutedMoe | SoftMoe]:
        return [m for m in self.modules() if isinstance(m, RoutedMoe | SoftMoe)]

    def aux_loss(self) -> Tensor:
        losses = [m.aux_loss for m in self.moe_layers() if m.aux_loss is not None]
        if not losses:
            return self.position.new_zeros(())
        # Mean, not sum: the routing regularisers keep the same weight relative to the
        # chess losses whatever the number of MoE layers.
        return torch.stack(losses).mean()

    def update_balance(self) -> None:
        for module in self.moe_layers():
            if isinstance(module, RoutedMoe):
                module.update_balance()

    def set_export_mode(self, enabled: bool) -> None:
        for module in self.moe_layers():
            if isinstance(module, RoutedMoe):
                module.export_mode = enabled


class EngineWrapper(nn.Module):
    """ONNX-facing module: WDL as probabilities, everything else unchanged."""

    def __init__(self, model: MouseMoe) -> None:
        super().__init__()
        self.model = model

    def forward(self, planes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        policy, wdl_logits, mlh = self.model(planes)
        return policy, wdl_logits.float().softmax(dim=-1), mlh


class HalfEngineWrapper(nn.Module):
    """EngineWrapper for a float16 model behind the engine's float32 tensors (FP16 export)."""

    def __init__(self, model: MouseMoe) -> None:
        super().__init__()
        self.model = model

    def forward(self, planes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        policy, wdl_logits, mlh = self.model(planes.half())
        return policy.float(), wdl_logits.float().softmax(dim=-1), mlh.float()


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def active_parameters_per_token(cfg: ModelConfig) -> int:
    """Body parameters a single token passes through (excludes per-board GAB and heads)."""
    d = cfg.d_model
    attention = 4 * d * d + d
    dense = 2 * d * cfg.dense_hidden + cfg.dense_hidden + d
    total = 0
    for index in range(cfg.layers):
        total += attention
        moe_here = cfg.ffn == "moe" and index % cfg.moe.every == cfg.moe.every - 1
        if not moe_here:
            total += dense
            continue
        expert = 2 * d * cfg.moe.expert_hidden + cfg.moe.expert_hidden + d
        shared = (
            2 * d * cfg.moe.shared_hidden + cfg.moe.shared_hidden + d
            if cfg.moe.shared_hidden
            else 0
        )
        total += shared + cfg.moe.top_k * expert + d * cfg.moe.experts
    return total


if POLICY_SIZE != 1858:
    raise AssertionError("policy head assumes Lc0's 1858-move output")
