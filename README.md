# Mouse-MoE

The network architecture and training code behind the nets of the
[Mouse](https://github.com/NoHashJustDash/Mouse) chess engine: transformers over the 64
squares with a mixture of experts in every layer, taking Lc0's input planes and producing
Lc0's policy, value and moves-left outputs.

M2.1, the current net, has **19.9M parameters, of which 8.4M run for each square**: every
layer sends each square to 2 of its 16 experts, plus a shared expert. Lc0's T1 net has
20.2M parameters and runs all of them. Mouse's nets learn mostly from the evaluations of
Lc0's much larger BT4 net on Lc0's open training games, with no self-play and no
reinforcement learning, on a single RTX 5070.

## Usage

Before you start, read the [docs](docs/0-contents.md). They cover setup, data,
training, scaling, export and testing.

Most people clone the repository, check their setup with the tiny example, then copy one
of the examples and edit it:

- [`tiny`](examples/tiny.toml): a 0.6M net that trains in minutes, to check that your
  setup works.
- [`small`](examples/small.toml): a 6.9M net on the games' own targets. No labelling needed,
  so it's quick to try ideas with.
- [`default`](examples/default.toml): **the template.** M2.1's network and first training
  stage, with every setting commented.
- [`large`](examples/large.toml): a 34.8M net within the engine's limits, to show how to
  scale up. Never trained.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1
.venv\Scripts\python.exe -I -m mouse_moe.download --config examples\tiny.toml
.venv\Scripts\python.exe -I -m mouse_moe.train --config examples\tiny.toml
```

Each release of Mouse also publishes M2.1's training checkpoint, so you can fine-tune from
a strong net (`--init-from`) instead of starting from zero
([docs/4-training.md](docs/4-training.md#continuing-forking-and-fine-tuning)).

## The nets

| Net | Configs | Result at 10+0.1 |
|---|---|---|
| M1 | [`configs/m1/`](configs/m1) | The first net: 256 wide, 13.2M parameters |
| M1.5 | [`configs/m20/`](configs/m20) | 19.9M parameters; −38.4 ± 15.1 Elo against lc0 + T1 |
| M2.0 | [`m2.toml`](configs/m2/m2.toml), [`m2_stage2.toml`](configs/m2/m2_stage2.toml) | +27.2 ± 12.4 Elo over M1.5; −12.3 ± 9.9 against lc0 + T1 |
| M2.1 | M2.0, then [`m2_stage3.toml`](configs/m2/m2_stage3.toml) | M2.0 trained on from 750k to 900k steps |

Conditions and every match: [docs/7-results.md](docs/7-results.md). Training M2.1 from
scratch takes about 40 hours on an RTX 5070, plus about 16 hours of BT4 labelling.

## Layout

| Path | Contents |
|---|---|
| `examples/` | Configs to copy and edit |
| `configs/` | The exact configs of every released net and of the comparison runs behind the design |
| `src/mouse_moe/` | Model, download, data loading, BT4 labelling, training, export |
| `docs/` | The guide |
| `manifests/` | Sizes and SHA-256 hashes of the Lc0 files used |
| `tests/` | Unit tests, plus checks against real data, TensorRT and lc0 |
| `scripts/` | Setup, benchmark and training wrappers for PowerShell |

## Help and feedback

Open an issue for bug reports and feature requests. When you report a training problem,
include your config and the last lines of `runs\<name>\metrics.jsonl`. If you train a
stronger net, submit it: [CONTRIBUTING.md](CONTRIBUTING.md) says how, and the
[export and test guide](docs/6-export-and-test.md) shows how to measure one.

## Made with AI assistance

The code, tests and documentation in this repository were written by AI coding
assistants, mainly Anthropic's Claude. The project's author directed the work: set the
goals, made the design decisions, and ran the training and the matches.

Nothing was taken on an assistant's word. The input encoding matches the Mouse engine's
bit for bit, and the engine's is checked against lc0. Every export is compared with
PyTorch before it is kept, and every net is measured in games before it is called
better.

## Licence

GNU General Public License, version 3 or later ([LICENSE](LICENSE)).

Nets trained with this code on Lc0's data contain information from the Leela Chess Zero
training data, which is made available under the Open Database License (ODbL). See
[docs/3-data.md](docs/3-data.md).

## Credits

- [Lc0 (Leela Chess Zero)](https://lczero.org/): the input encoding, the policy map, the
  training data, the smolgen attention bias, and the BT4 net whose evaluations are the
  training targets. Mouse-MoE is not affiliated with or endorsed by the Lc0 project.
- Monroe et al., *Chessformer*: the square-token transformer, its attention bias and its
  policy head.
- DeepSeek-AI and Wang et al.: the shared expert and auxiliary-loss-free balancing.
- [bullet](https://github.com/jw1912/bullet) by Jamie Whiting, whose examples-first layout
  this repository follows.
- The other papers in [docs/8-references.md](docs/8-references.md).
- [PyTorch](https://pytorch.org/), [Triton](https://triton-lang.org/),
  [ONNX](https://onnx.ai/), [ONNX Runtime](https://onnxruntime.ai/) and NVIDIA TensorRT.
