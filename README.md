# Mouse-MoE

Mouse-MoE is the neural-network architecture and training code behind the nets of the
[Mouse](https://github.com/NoHashJustDash/Mouse) chess engine. The nets are transformers
over the 64 squares with a mixture of experts in every layer. They take Lc0's input
planes and produce Lc0's policy, value and moves-left outputs.

The current shape has **19.9M parameters**, of which **8.4M run for each square**: every
layer routes each square to 2 of its 16 experts, plus a shared expert. Lc0's T1 net, for
comparison, has 20.2M parameters and runs all of them. The nets learn mostly from the
evaluations of Lc0's much larger BT4 net on Lc0's open training games. There is no
self-play and no reinforcement learning, and everything trains on a single RTX 5070.

## The nets

| Net | Training config | Result |
|---|---|---|
| M1 | [`configs/m1/`](configs/m1) | The first net: 256 wide, 13.2M parameters |
| M1.5 | [`configs/m20/`](configs/m20) | 19.9M parameters; −38.4 ± 15.1 Elo against lc0 + T1 |
| M2.0 | [`configs/m2/m2.toml`](configs/m2/m2.toml), [`m2_stage2.toml`](configs/m2/m2_stage2.toml) | +27.2 ± 12.4 Elo over M1.5; −12.3 ± 9.9 against lc0 + T1 |
| M2.1 | M2.0, then [`m2_stage3.toml`](configs/m2/m2_stage3.toml) | M2.0 trained on from 750k to 900k steps |

All results are from 10+0.1 matches on one machine, with the conditions in
[docs/RESULTS.md](docs/RESULTS.md). The trained nets ship with the Mouse engine.

## Documentation

- [Architecture](docs/ARCHITECTURE.md): the layers, routing and parameter counts.
- [Training](docs/TRAINING.md): setup, labelling, the three training stages and export.
- [Data](docs/DATA.md): which Lc0 files are used, how they are checked, and the licence.
- [Results](docs/RESULTS.md): every match, with conditions.
- [Scaling up](docs/SCALING.md): what to change for a larger net, and the engine's limits.
- [References](docs/REFERENCES.md): the papers and projects each idea comes from.

## Quick start

On Windows with an NVIDIA GPU and python.org Python 3.12:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Then follow [docs/TRAINING.md](docs/TRAINING.md). Training M2.1 from scratch takes about
40 hours on an RTX 5070, plus about 16 hours to label the data with BT4.

## Layout

| Path | Contents |
|---|---|
| `src/mouse_moe/` | Model, data loading, training, BT4 labelling, export |
| `configs/` | Every net's training config, and the comparison runs behind the design |
| `tests/` | Unit tests, plus checks against real data, TensorRT and lc0 |
| `manifests/` | Sizes and SHA-256 hashes of the Lc0 training files used |
| `scripts/` | Setup, benchmark and training wrappers for PowerShell |

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
[docs/DATA.md](docs/DATA.md).

## Credits

- [Lc0 (Leela Chess Zero)](https://lczero.org/): the input encoding, the policy map, the
  training data, the smolgen attention bias, and the BT4 net whose evaluations are the
  training targets. Mouse-MoE is not affiliated with or endorsed by the Lc0 project.
- Monroe et al., *Chessformer*: the square-token transformer, its attention bias and its
  policy head.
- DeepSeek-AI and Wang et al.: the shared expert and auxiliary-loss-free balancing.
- The other papers in [docs/REFERENCES.md](docs/REFERENCES.md).
- [PyTorch](https://pytorch.org/), [Triton](https://triton-lang.org/),
  [ONNX](https://onnx.ai/), [ONNX Runtime](https://onnxruntime.ai/) and NVIDIA TensorRT.
