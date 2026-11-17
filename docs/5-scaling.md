# Scaling up

M2.1 is sized for one 12 GB GPU and a few days of training. Everything that sets its size
is in the config, so a larger net needs a new config, more data, and more GPU time.
Bigger nets cost search speed too, so measure every change in games.

## What you can change

All of these are in a config's `[model]` and `[model.moe]` sections:

| Setting | M2.1 | What it changes |
|---|---:|---|
| `d_model` | 320 | Width of every token. Cost grows with its square. |
| `layers` | 8 | Number of encoder layers. Cost grows linearly. |
| `heads` | 10 | Attention heads. `d_model` must divide evenly by it. |
| `experts` | 16 | Routed experts per layer. More experts add parameters without adding compute per square. |
| `top_k` | 2 | Experts each square uses. Compute per square grows with it. |
| `expert_hidden` | 160 | Hidden width of each expert. |
| `shared_hidden` | 320 | Hidden width of the shared expert (0 for none). |
| `[model.gab]` | 16 / 64 / 64 | Size of the attention-bias generator: `compress`, `hidden`, `per_head`. |

[`mouse_moe.budget`](../src/mouse_moe/budget.py) gives a config's exact parameter count
(`parameter_breakdown`, `parameter_total`) and its multiply-adds per position
(`macs_per_position`) without building the model.
[`mouse_moe.benchmark`](../src/mouse_moe/benchmark.py) measures training speed and GPU
memory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\benchmark.ps1 -Config configs\<your config>.toml
```

## Limits for running a net in the Mouse engine

The engine runs these nets on TensorRT, and each layer's experts go through its
`mouse_moe_trt` plugin (`engine/nn/trt_moe/moe_plugin.cu` in the
[Mouse repository](https://github.com/NoHashJustDash/Mouse)). `mouse_moe.export_plugin`
refuses a net the plugin can't run:

- **Token routing only.** No soft MoE and no per-position routing.
- **At most 16 experts** per layer, and **top-k of at most 4**.
- **`d_model` and `expert_hidden` must be divisible by 8.**
- **Width is limited by the router's shared memory.** The router keeps every expert's
  weights in shared memory: with 16 experts, that is about 128 bytes × `d_model`. Most RTX
  GPUs allow 99 KB per block, which is enough for `d_model` up to about 768. Wider nets
  need fewer experts or a change to the plugin.

To go past these limits, extend the plugin and its tests in the Mouse repository first.
The plugin-free export (`mouse_moe.export_onnx`) has none of these limits. It runs on
ONNX Runtime, but much more slowly.

## Data

Larger nets need more positions, or they repeat the same ones too often.

- **M2.1:** about 297M positions (182M test80, 115M test90), seen about three times each
  over 900k steps.
- **More data:** Lc0 publishes far more than this. Add hours to a config's `train_shards`
  and `game_shards`, give their hashes in `manifests/`, and label them with
  `mouse_moe.label_teacher`.
- **Labelling cost:** about 5,000 positions per second on an RTX 5070, or about
  5.5 hours per 100M positions.
- **Keep the held-out hours out of training,** so validation and test results stay
  honest.

## GPU memory and time

Speeds measured on one RTX 5070 (12 GB), batch 1,024, bfloat16 with `torch.compile`:

| Shape | Positions/s | Peak GPU memory |
|---|---:|---:|
| M2.1: 320 × 8 layers, 16 experts × 160, top 2 | 6,216 | 7.1 GiB |
| 320 × 10 layers, 8 experts × 256, top 2 | 4,976 | – |
| 320 × 10 layers, 16 experts × 128, top 4 | 1,276 | 10.7 GiB |
| 384 × 12 layers, 46M (`configs/m1_46m.toml`), batch 512, measured before the faster expert dispatch | 1,359 | 10.2 GB |

When a batch of 1,024 doesn't fit, set `grad_accum_steps` under `[train]`: each step then
adds up the gradients of several smaller micro-batches. The training recipe stays the
same.

Training caps its own host and GPU memory (see [4-training.md](4-training.md)), so a net that
doesn't fit stops with an out-of-memory error and doesn't slow the machine down.

## Search speed

Engine strength depends on how many positions it searches in its time, not only on the
net's quality. M2.0, which has the same shape as M2.1, uses 8.4M parameters per square
against T1's 20.2M, yet searched only about 14% faster than lc0 with T1. A net that is
better per position but much slower can still lose games. Compare nets in matches at the
time control you care about.
