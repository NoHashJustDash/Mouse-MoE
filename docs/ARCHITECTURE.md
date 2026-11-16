# Architecture

Mouse-MoE nets are encoder-only transformers over the 64 squares of the board, with a
mixture of experts (MoE) in place of each layer's feed-forward block. The model is
`MouseMoe` in [`src/mouse_moe/model.py`](../src/mouse_moe/model.py); every size below
comes from a config under [`configs/`](../configs).

This page describes the shape shared by M1.5, M2.0 and M2.1
([`configs/m2/m2_stage3.toml`](../configs/m2/m2_stage3.toml)).

## Input and output

The interface is Lc0's, so the [Mouse engine](https://github.com/NoHashJustDash/Mouse) runs
these nets the same way it runs Lc0's:

- **Input:** Lc0's 112 input planes (input format 1: the last eight positions, castling
  rights, side to move and the fifty-move counter), shape `[batch, 112, 8, 8]`.
- **Policy:** logits over Lc0's 1,858 moves.
- **Value:** win, draw and loss probabilities.
- **Moves left:** the expected number of plies until the game ends.

## Layers

1. **Embedding.** Each square becomes a token: its 112 plane values go through a linear
   layer to width 320, a learned position vector for the square is added, then Mish and
   a layer norm.
2. **Eight encoder layers**, each with:
   - **Attention:** 10 heads of width 32 over the 64 tokens.
   - **Geometric attention bias (Lc0's "smolgen"):** each layer compresses the board to
     16 values per square, summarises the whole board in 64 values, and turns that
     summary into 64 values per head. A projection shared by all layers expands each
     head's 64 values into a 64 × 64 bias, which is added to that head's attention
     logits. Attention can then depend on the whole position, not only on how similar
     two squares are.
   - **Mixture-of-experts feed-forward:** a router scores 16 experts for every square
     and sends it to its top 2. Each expert is a two-layer Mish network, 320 → 160 → 320.
     The two outputs are weighted by the router's scores, renormalised over the two
     chosen experts. A shared expert (320 → 320 → 320) runs on every square and is
     added to them.
   - **Post-layer norm with DeepNet residual scaling** (Wang et al. 2022) around both
     blocks.
3. **Heads:**
   - **Policy:** attention scores from each from-square to each to-square, plus four
     promotion outputs, gathered into Lc0's 1,858-move order.
   - **Value:** per-square projection to 32, flattened, then 2,048 → 128 → 3.
   - **Moves left:** per-square projection to 8, flattened, then 512 → 128 → 1, with a
     softplus so the output is never negative.

## Routing and balance

- **Token routing:** each square chooses its own experts, so one position uses many
  experts across its 64 squares.
- **Auxiliary-loss-free balancing** (Wang et al. 2024, arXiv:2408.15664): each expert has
  a bias, added to the router's scores only when choosing experts, never when weighting
  them. After each step the biases of overloaded experts go down and those of
  underloaded experts go up. A small balance loss (weight 1e-4) and a router z-loss
  (weight 1e-3) are added as well.
- **After training,** the biases are re-tuned for the averaged (SWA) weights on 300
  batches with the weights frozen, because the live biases were tuned for the live
  router.
- **The exported router runs in float32,** so half precision can't change which
  experts a square chooses.

## Parameters

| Component (all 8 layers) | Parameters |
|---|---:|
| Embedding and position vectors | 56,960 |
| Attention (q, k, v, output) | 3,279,360 |
| Geometric attention bias, including the shared 64 → 4,096 projection | 1,161,728 |
| Layer norms | 5,120 |
| Routers | 40,960 |
| **Routed experts** | **13,168,640** |
| Shared experts | 1,643,520 |
| Policy head | 206,724 |
| Value head | 272,931 |
| Moves-left head | 68,361 |
| **Total** | **19,904,304** |
| **Used for each square** (2 of the 16 experts) | **8,381,744** |

For comparison, Lc0's T1-256x10-distilled-swa-2432500 has 20,198,188 parameters, and all of
them run on every position.

## Running it in the engine

[`mouse_moe.export_plugin`](../src/mouse_moe/export_plugin.py) writes each layer's experts
as one `MouseMoeExperts` node. That node is a TensorRT plugin in the Mouse engine
(`engine/nn/trt_moe/moe_plugin.cu`), which does the routing and both expert layers with
CUTLASS grouped matrix multiplies. Everything else is standard ONNX, opset 17.

The export is built in FP16 with TensorRT and compared with PyTorch on real validation
positions. It is deleted unless at least 97% of top moves agree, the mean policy KL
divergence is below 5e-4 and no win/draw/loss output differs by more than 0.15. For
M2.1: 99.2% top-move agreement, mean KL 1.2e-4, largest win/draw/loss error 0.010.

[`mouse_moe.export_onnx`](../src/mouse_moe/export_onnx.py) writes a plugin-free version
using only standard ONNX operators. It runs on ONNX Runtime's CPU provider but is much
slower on the GPU.

## Earlier shapes

| Net | Config | Width × layers | Experts | Total / used |
|---|---|---|---|---:|
| M1 | `configs/m1/` | 256 × 8 | 16, top 2, shared | 13.2M / 5.9M |
| M1.5, M2.0, M2.1 | `configs/m20/`, `configs/m2/` | 320 × 8 | 16, top 2, shared | 19.9M / 8.4M |

`configs/ablation/` holds the short comparison runs behind these choices, all at M1's
size:

| Config | Variant |
|---|---|
| `ab_d0` | Dense, same compute per square as M1 (~5.8M) |
| `ab_d1` | Dense, same total parameters as M1 (~13.2M) |
| `ab_m1` | M1: token routing, shared expert + 16 routed, top 2 |
| `ab_m2` | One expert choice per board instead of per square |
| `ab_m3` | Soft MoE, 4 slots per expert |
| `ab_m4` | MoE in every other layer only |
| `ab_m5` | 8 wider experts, top 1 (same compute as M1) |

`configs/deployment/` compares depth (8 and 12 layers) and a dense control at full width.
`configs/m1_46m.toml` is the first, larger 46M design. It was never trained to release:
on one RTX 5070 it trained at 1,359 positions per second against 6,216 for the 20M shape.
