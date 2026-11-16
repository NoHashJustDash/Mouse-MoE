# SPDX-License-Identifier: GPL-3.0-or-later
"""Model, routing and loss tests (marker: torch)."""

from dataclasses import replace

import numpy as np
import pytest

from conftest import ROOT, make_record

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch", reason="PyTorch is installed by scripts/setup.ps1")

from mouse_moe.budget import parameter_total  # noqa: E402
from mouse_moe.config import GabConfig, ModelConfig, MoeConfig, load_config  # noqa: E402
from mouse_moe.losses import compute_losses, masked_log_policy  # noqa: E402
from mouse_moe.model import (  # noqa: E402
    EXPORT_TILE_ROWS,
    EngineWrapper,
    MouseMoe,
    RoutedMoe,
    expand_planes,
    parameter_count,
    top_k_ids_without_index_ops,
)
from mouse_moe.policy_map import policy_index_of  # noqa: E402
from mouse_moe.v6 import expand_planes as expand_planes_np  # noqa: E402
from mouse_moe.v6 import parse_records, records_to_batch  # noqa: E402

TINY = ModelConfig(
    d_model=64,
    layers=2,
    heads=4,
    dense_hidden=128,
    gab=GabConfig(compress=8, hidden=32, per_head=32),
    moe=MoeConfig(experts=4, top_k=2, expert_hidden=32, shared_hidden=64),
)


def _planes(count: int) -> "torch.Tensor":
    batch = records_to_batch(parse_records(make_record().tobytes() * count))
    return torch.from_numpy(expand_planes_np(batch.packed, batch.aux))


@pytest.mark.parametrize(
    "path", sorted((ROOT / "configs").rglob("*.toml")), ids=lambda p: p.stem
)
def test_parameter_count_matches_budget_module(path) -> None:  # type: ignore[no-untyped-def]
    config = load_config(path)
    assert parameter_count(MouseMoe(config.model)) == parameter_total(config.model)


@pytest.mark.parametrize("routing", ["token", "position", "soft"])
@pytest.mark.parametrize("every", [1, 2])
def test_forward_contract(routing: str, every: int) -> None:
    cfg = replace(TINY, moe=replace(TINY.moe, routing=routing, every=every))
    model = MouseMoe(cfg).eval()
    policy, wdl, mlh = EngineWrapper(model)(_planes(3))
    assert policy.shape == (3, 1858) and wdl.shape == (3, 3) and mlh.shape == (3, 1)
    assert torch.allclose(wdl.sum(dim=1), torch.ones(3), atol=1e-5)
    assert (mlh >= 0).all() and torch.isfinite(policy).all()


def test_dense_model_has_no_moe() -> None:
    model = MouseMoe(replace(TINY, ffn="dense"))
    assert model.moe_layers() == []
    assert model.aux_loss().item() == 0.0


def test_gpu_plane_expansion_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    packed = rng.integers(0, 256, size=(5, 832), dtype=np.uint8)
    aux = rng.random((5, 8), dtype=np.float32)
    expected = expand_planes_np(packed, aux)
    actual = expand_planes(torch.from_numpy(packed), torch.from_numpy(aux)).numpy()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("routing", ["token", "position"])
def test_export_dispatch_matches_training_dispatch(routing: str) -> None:
    torch.manual_seed(0)
    cfg = replace(TINY, moe=replace(TINY.moe, routing=routing))
    model = MouseMoe(cfg).eval()
    planes = _planes(4) + 0.1 * torch.randn(4, 112, 8, 8)
    with torch.no_grad():
        reference = model(planes)
        model.set_export_mode(True)
        exported = model(planes)
    for a, b in zip(reference, exported, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=0)


@pytest.mark.parametrize("routing", ["token", "position"])
def test_training_dispatch_gradients_match_export_reference(routing: str) -> None:
    # The sorted training dispatch must train exactly like the ONNX export
    # path, not only agree with it in the forward pass.
    torch.manual_seed(0)
    cfg = replace(TINY, moe=replace(TINY.moe, routing=routing))
    model = MouseMoe(cfg).train()
    planes = _planes(4) + 0.1 * torch.randn(4, 112, 8, 8)
    grads = []
    for export in (False, True):
        model.set_export_mode(export)
        model.zero_grad(set_to_none=True)
        policy, wdl, mlh = model(planes)
        (policy.square().mean() + wdl.square().mean() + mlh.square().mean()).backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    model.set_export_mode(False)
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        torch.testing.assert_close(grads[0][name], grads[1][name], atol=1e-5, rtol=1e-4, msg=name)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_index_free_top_k_selects_the_same_experts_as_topk(k: int) -> None:
    torch.manual_seed(k)
    scores = torch.randn(512, 16)
    expected = torch.topk(scores, k, dim=-1).indices.sort(dim=-1).values
    torch.testing.assert_close(top_k_ids_without_index_ops(scores, k), expected)


def test_index_free_top_k_breaks_exact_ties_toward_low_columns() -> None:
    scores = torch.tensor([[0.5, 0.5, 0.5, 0.1], [0.2, 0.9, 0.9, 0.9], [0.3, 0.3, 0.3, 0.3]])
    ids = top_k_ids_without_index_ops(scores, 2)
    assert ids.tolist() == [[0, 1], [1, 2], [0, 1]]


def test_index_free_top_k_keeps_a_higher_column_above_a_tie() -> None:
    # ab_kd_ft layer scores 0.5918 (experts 3, 14) tied under 1.1835 (expert 15): the
    # tie must fill only the places left after every strictly higher column.
    scores = torch.tensor([[0.5, 0.5, 0.9, 0.1], [0.4, 0.4, 0.4, 0.9]])
    assert top_k_ids_without_index_ops(scores, 2).tolist() == [[0, 2], [0, 3]]
    wide = torch.tensor([[0.4, 0.4, 0.4, 0.8, 0.9]])
    assert top_k_ids_without_index_ops(wide, 3).tolist() == [[0, 3, 4]]


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("tokens", [1, 64, EXPORT_TILE_ROWS, 5 * 64, 9 * 64])
def test_tiled_export_dispatch_is_exact(tokens: int, skew: bool) -> None:
    # Skewed: every token picks experts 0 and 1, as one board can in a real net (ab_m1:
    # max/mean expert load 8.0). With tokens == EXPORT_TILE_ROWS each expert fills
    # exactly one tile, the boundary case for the padded slot layout.
    torch.manual_seed(tokens)
    layer = RoutedMoe(TINY.d_model, TINY.moe, 1.0, position=False).eval()
    flat = torch.randn(tokens, TINY.d_model)
    if skew:
        selected = torch.tensor([[0, 1]]).expand(tokens, 2).contiguous()
    else:
        selected = torch.rand(tokens, TINY.moe.experts).argsort(dim=-1)[:, :2]
    gates = torch.rand(tokens, 2)
    gates = gates / gates.sum(dim=-1, keepdim=True)
    with torch.no_grad():
        expected = layer._dispatch_grouped(flat, selected, gates)
        actual = layer._dispatch_export(flat, selected, gates)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def _per_token_reference(layer: RoutedMoe, flat: "torch.Tensor", selected, gates):  # type: ignore[no-untyped-def]
    """Each token through each chosen expert, one at a time: the dispatch's definition."""
    e = layer.experts
    rows = []
    for t in range(flat.shape[0]):
        out = torch.zeros(flat.shape[1])
        for j in range(selected.shape[1]):
            i = int(selected[t, j])
            hidden = torch.nn.functional.mish(flat[t] @ e.w1[i] + e.b1[i])
            out = out + gates[t, j] * (hidden @ e.w2[i] + e.b2[i])
        rows.append(out)
    return torch.stack(rows)


@pytest.mark.parametrize("skew", [False, True])
def test_grouped_dispatch_matches_per_token_experts_in_value_and_gradient(skew: bool) -> None:
    torch.manual_seed(1)
    layer = RoutedMoe(TINY.d_model, TINY.moe, 1.0, position=False)
    tokens = 48
    flat = torch.randn(tokens, TINY.d_model, requires_grad=True)
    if skew:  # experts 2 and 3 get nothing: empty groups
        selected = torch.tensor([[0, 1]]).expand(tokens, 2).contiguous()
    else:
        selected = torch.rand(tokens, TINY.moe.experts).argsort(dim=-1)[:, :2]
    gates = torch.rand(tokens, 2)
    gates = gates / gates.sum(dim=-1, keepdim=True)
    reference = _per_token_reference(layer, flat, selected, gates)
    want = torch.autograd.grad(reference.square().sum(), [flat, layer.experts.b1, layer.experts.w2])
    actual = layer._dispatch_grouped(flat, selected, gates)
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
    got = torch.autograd.grad(actual.square().sum(), [flat, layer.experts.b1, layer.experts.w2])
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    # bf16 autocast takes the grouped-matmul kernel; it agrees to bf16 rounding.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = layer._dispatch_grouped(flat, selected, gates)
    assert mixed.dtype == torch.float32
    torch.testing.assert_close(mixed, reference, atol=0.05, rtol=0.05)


def test_bf16_training_layer_traces_whole_without_host_syncs() -> None:
    # fullgraph=True fails on any graph break, e.g. reading expert counts back to the host.
    torch.manual_seed(2)
    layer = RoutedMoe(TINY.d_model, TINY.moe, 1.0, position=False).train()
    traced = torch.compile(layer, fullgraph=True, backend="eager")
    x = torch.randn(2, 64, TINY.d_model)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        traced(x).float().sum().backward()
    torch._dynamo.reset()


def test_position_routing_sends_whole_board_to_same_experts() -> None:
    cfg = replace(TINY, moe=replace(TINY.moe, routing="position"))
    layer = next(m for m in MouseMoe(cfg).modules() if isinstance(m, RoutedMoe))
    x = torch.randn(3, 64, cfg.d_model)
    _, _, selected, gates = layer._route(x)
    per_board = selected.reshape(3, 64, -1)
    assert (per_board == per_board[:, :1]).all()
    assert torch.allclose(gates.sum(dim=1), torch.ones(3 * 64))


@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("routing", ["token", "position"])
def test_router_learns_from_the_task_loss_alone(top_k: int, routing: str) -> None:
    # Regression: top-1 gates were renormalised to p/p = 1, so the chess loss sent
    # the router a zero gradient (measured 1.9e-8 vs 8.0e-2 at top-2).
    torch.manual_seed(0)
    cfg = replace(
        TINY.moe,
        experts=4,
        top_k=top_k,
        shared_hidden=0,
        routing=routing,
        aux_loss_weight=0.0,
        z_loss_weight=0.0,
    )
    layer = RoutedMoe(TINY.d_model, cfg, 1.0, position=routing == "position").train()
    layer(torch.randn(3, 64, TINY.d_model)).square().mean().backward()
    assert layer.router.weight.grad is not None
    assert layer.router.weight.grad.norm() > 1e-4


def test_balance_bias_moves_toward_underloaded_experts() -> None:
    layer = next(m for m in MouseMoe(TINY).modules() if isinstance(m, RoutedMoe))
    layer.load.copy_(torch.tensor([10.0, 0.0, 5.0, 5.0]))
    layer.update_balance()
    rate = TINY.moe.bias_update_rate
    assert torch.allclose(layer.balance_bias, torch.tensor([-rate, rate, 0.0, 0.0]))
    assert layer.load.sum() == 0


def test_balance_biases_stay_centred_without_changing_the_selection() -> None:
    # One overloaded expert moves three biases up and one down; uncentred, all four
    # drift up together over a run (m20: +4 to +7 after 500k steps).
    layer = next(m for m in MouseMoe(TINY).modules() if isinstance(m, RoutedMoe))
    rate = TINY.moe.bias_update_rate
    layer.balance_bias.copy_(torch.tensor([5.0, 5.1, 4.9, 5.0]))
    x = torch.randn(2, 64, TINY.d_model)
    before = layer._scores(x)[2].topk(TINY.moe.top_k, dim=-1).indices
    layer.load.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0]))
    layer.update_balance()  # nothing routed: only the centring applies
    torch.testing.assert_close(layer.balance_bias, torch.tensor([0.0, 0.1, -0.1, 0.0]))
    after = layer._scores(x)[2].topk(TINY.moe.top_k, dim=-1).indices
    assert torch.equal(before, after)
    layer.load.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
    layer.update_balance()
    expected = torch.tensor([-rate, 0.1 + rate, -0.1 + rate, rate]) - rate / 2
    torch.testing.assert_close(layer.balance_bias, expected)
    assert abs(float(layer.balance_bias.mean())) < 1e-7


def test_policy_head_gather_uses_source_destination_logits() -> None:
    model = MouseMoe(TINY).eval()
    head = model.policy
    x = torch.randn(2, 64, TINY.d_model)
    out = head(x)
    q, k = head.query(x), head.key(x)
    moves = (q @ k.transpose(1, 2)) * head.scale
    promo = head.promotion(k[:, 56:64, :])
    index = policy_index_of()
    torch.testing.assert_close(out[:, index["e2e4"]], moves[:, 12, 28])
    torch.testing.assert_close(out[:, index["a7a8q"]], moves[:, 48, 56] + promo[:, 0, 0])
    torch.testing.assert_close(out[:, index["a7a8"]], moves[:, 48, 56] + promo[:, 0, 3])


def test_masked_policy_ignores_illegal_moves() -> None:
    target = torch.full((1, 1858), -1.0)
    target[0, :3] = torch.tensor([0.5, 0.25, 0.25])
    logits = torch.zeros(1, 1858)
    logits[0, 100] = 50.0  # an illegal move with a huge logit must not matter
    logp = masked_log_policy(logits, target)
    torch.testing.assert_close(logp[0, :3].exp().sum(), torch.tensor(1.0))
    assert torch.all(logp[0, 3:] == 0)


def test_tiny_model_overfits_one_batch() -> None:
    torch.manual_seed(0)
    model = MouseMoe(TINY)
    batch_np = records_to_batch(parse_records(make_record().tobytes() * 16))
    batch = {
        "policy": torch.from_numpy(batch_np.policy),
        "wdl": torch.from_numpy(batch_np.wdl),
        "q_wdl": torch.from_numpy(batch_np.q_wdl),
        "plies_left": torch.from_numpy(batch_np.plies_left),
    }
    planes = expand_planes(torch.from_numpy(batch_np.packed), torch.from_numpy(batch_np.aux))
    optimizer = torch.optim.NAdam(model.parameters(), lr=3e-3)
    weights = load_config(ROOT / "configs" / "ablation" / "ab_m1.toml").loss

    # Soft targets have irreducible cross-entropy (their own entropy); progress is
    # measured as the fraction of the gap to that floor that training removes.
    def entropy(p: "torch.Tensor") -> float:
        p = p[p > 0]
        return float(-(p * p.log()).sum())

    floor = (
        weights.policy * entropy(batch["policy"][0])
        + weights.wdl * entropy(batch["wdl"][0])
        + weights.q_wdl * entropy(batch["q_wdl"][0])
    )
    losses = []
    for _ in range(80):
        terms = compute_losses(model(planes), batch, weights, model.aux_loss())
        optimizer.zero_grad()
        terms.total.backward()
        optimizer.step()
        model.update_balance()
        losses.append(terms.total.item())
    assert losses[0] > floor
    assert losses[-1] - floor < 0.25 * (losses[0] - floor)


def test_aux_loss_is_the_mean_over_moe_layers() -> None:
    # Regression: the per-layer routing losses were summed, so a 12-layer net got
    # 1.5x the routing regularisation of an 8-layer one.
    model = MouseMoe(TINY).train()
    model(_planes(2))
    layer_losses = [m.aux_loss for m in model.moe_layers()]
    assert len(layer_losses) == TINY.layers
    torch.testing.assert_close(model.aux_loss(), torch.stack(layer_losses).mean())


@pytest.mark.parametrize("routing", ["token", "position"])
def test_gates_are_the_renormalised_scores_of_the_chosen_experts(routing: str) -> None:
    torch.manual_seed(0)
    cfg = replace(TINY.moe, routing=routing)
    layer = RoutedMoe(TINY.d_model, cfg, 1.0, position=routing == "position")
    layer.balance_bias.copy_(torch.tensor([0.3, -0.1, 0.2, 0.0]))
    _, scores, selected, gates = layer._route(torch.randn(2, 64, TINY.d_model))
    if routing == "position":
        scores = scores.repeat_interleave(64, dim=0)
    chosen = scores.gather(1, selected)
    torch.testing.assert_close(gates, chosen / chosen.sum(dim=1, keepdim=True))
