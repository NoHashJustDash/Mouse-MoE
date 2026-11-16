# Getting started

This gets you from a fresh clone to a trained (tiny) net in under an hour. Everything was
built and tested on Windows 11 with one NVIDIA GPU (an RTX 5070, 12 GB); see
[Platforms](#platforms) for other systems. Run every command from the repository root.

## 1. Install

You need:

- An NVIDIA GPU with a recent driver. PyTorch brings its own CUDA runtime.
- [python.org Python 3.12](https://www.python.org/downloads/release/python-31210/).
- About 60 GB of free disk for the full M2.1 data, or 5 GB for the tiny example.

Then:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1
```

This creates `.venv`, installs PyTorch 2.9.1 for CUDA 12.8 and the pinned tools in
`requirements-train.txt`, checks that PyTorch supports your GPU, and runs lint, type checks
and the tests that need nothing downloaded. It fails if any of them fails or is skipped.

Every command below uses that environment, as `.venv\Scripts\python.exe -I`. The `-I`
flag (isolated mode) is required: training data is untrusted input, so the tools refuse
to run without it.

## 2. Check your setup with the tiny example

```powershell
.venv\Scripts\python.exe -I -m mouse_moe.download --config examples\tiny.toml
.venv\Scripts\python.exe -I -m mouse_moe.train --config examples\tiny.toml
```

- **The download** fetches two Lc0 test80 files (2.7 GiB), checks each one's size and
  SHA-256, and writes `data\test80\MANIFEST.csv`.
- **The training run** goes for 2,000 steps. Its log, checkpoints and validation results
  are in `runs\tiny\`.

If both finish, your setup works.

## 3. Pick an example and edit it

Most people copy one of the examples and change it:

| Example | What it is | Needs |
|---|---|---|
| [`tiny.toml`](../examples/tiny.toml) | 0.6M parameters, minutes; checks the setup | 2 data files |
| [`small.toml`](../examples/small.toml) | 6.9M parameters, trained on the games' own targets | the 22 test80 files, no labelling |
| [`default.toml`](../examples/default.toml) | **The template.** M2.1's network and first training stage, every setting commented | all 77 files and BT4 labels |
| [`large.toml`](../examples/large.toml) | 34.8M parameters, within the engine's limits; **never trained** | as `default.toml`, plus more GPU time |

`configs/` holds the exact configs of every released net and of the comparison runs behind
the design. Leave those unchanged so results can be reproduced, and make your own in
`examples/` or anywhere else.

## 4. Optional tools

Only labelling, exporting and the full test run need these:

| Tool | Get it from | Set |
|---|---|---|
| TensorRT 10.9.0.34 | NVIDIA. Install its Python wheel into `.venv`: `.venv\Scripts\python.exe -I -m pip install <TensorRT>\python\tensorrt-10.9.0.34-cp312-none-win_amd64.whl` | `MOUSE_MOE_TENSORRT_LIB` = `<TensorRT>\lib` |
| lc0 v0.32.1 | [lczero.org](https://lczero.org/) | `MOUSE_MOE_LC0` = `lc0.exe` (search-labelling test only) |
| BT4 weights | `BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz` from Lc0 | `MOUSE_MOE_BT4` (search-labelling test only) |
| `mouse_moe_trt.dll` | Build the [Mouse engine](https://github.com/NoHashJustDash/Mouse) with its `windows-moe` CMake preset | `MOUSE_MOE_PLUGIN` (plugin export test) |

Labelling uses BT4 as ONNX, made with lc0's own converter:

```powershell
lc0.exe leela2onnx --input=BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --output=bt4-it332-f16.onnx --onnx-data-type=f16
```

With all four set, `scripts\setup.ps1 -Full` runs every test, including the ones that need
the downloaded data.

## Platforms

Training was only ever run on Windows. Two parts are Windows-specific:

- **The memory limits:** `mouse_moe.resources` uses Windows job objects.
- **The scripts:** PowerShell.

The model, data loading and export are plain PyTorch and Python, so a Linux port mainly
needs Linux versions of the memory limits. Contributions are welcome.
