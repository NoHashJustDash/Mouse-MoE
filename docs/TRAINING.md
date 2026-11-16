# Training a net

These steps reproduce M2.1 on Windows with one NVIDIA GPU. They were run on an RTX 5070
(12 GB). Run every command from the repository root. Each step stops with a clear error if
something is wrong; none of them falls back silently.

## 1. Environment

Install [python.org Python 3.12](https://www.python.org/downloads/release/python-31210/),
then:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1
```

This creates `.venv`, installs `torch==2.9.1+cu128` and the pinned tools in
`requirements-train.txt`, checks that PyTorch supports your GPU, then runs lint, type
checks and the tests.

- **Default:** the tests that need nothing downloaded.
- **`-Full`:** everything, including the tests that need the data, TensorRT, lc0 and
  BT4. These need the environment variables in the next section.

Either way, no selected test may be skipped.

## 2. Tools and environment variables

Only labelling, exporting and the full test run need these:

| Variable | Points to | Used by |
|---|---|---|
| `MOUSE_MOE_TENSORRT_LIB` | The `lib` folder of TensorRT 10.9.0.34, the version of the installed `tensorrt` wheel | `label_teacher`, `export_plugin` |
| `MOUSE_MOE_PLUGIN` | `mouse_moe_trt.dll`, built by the [Mouse engine](https://github.com/NoHashJustDash/Mouse)'s `windows-moe` CMake preset | plugin export test |
| `MOUSE_MOE_LC0` | `lc0.exe` | lc0 search-labelling test |
| `MOUSE_MOE_BT4` | `BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz` | lc0 search-labelling test |

Convert BT4 to ONNX for labelling with lc0's own converter:

```powershell
lc0.exe leela2onnx --input=BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --output=bt4-it332-f16.onnx --onnx-data-type=f16
```

## 3. Data

Download the files and check them as described in [DATA.md](DATA.md).

## 4. BT4 labels (about 16 hours on an RTX 5070)

`label_teacher` labels the training files listed in a config's `train_shards`, at about
5,000 positions per second. M2.1's labels come from three runs into two folders:

```powershell
.venv\Scripts\python.exe -I -m mouse_moe.label_teacher --onnx bt4-it332-f16.onnx --config configs\m2\m2.toml --out data\test80\labels\bt4-it332
.venv\Scripts\python.exe -I -m mouse_moe.label_teacher --onnx bt4-it332-f16.onnx --config configs\m2\label_test90.toml --out data\test90\labels\bt4-it332
.venv\Scripts\python.exe -I -m mouse_moe.label_teacher --onnx bt4-it332-f16.onnx --config configs\m20\stage3.toml --out data\test90\labels\bt4-it332
```

The first labels the 22 test80 hours (about 10 hours), the second 48 test90 hours (about
6 hours), the third the other 5 test90 hours. A stopped run resumes at the next
unfinished file.

## 5. Measure speed first

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\benchmark.ps1 -Config configs\m2\m2.toml
```

This prints training and data-loading speed, peak GPU memory and the estimated time for
the config. The GPU should be otherwise idle.

## 6. Train M2.1 in three stages

| Stage | Config | Steps | Targets | Learning rate | Time on an RTX 5070 |
|---|---|---|---|---|---|
| A | `configs/m2/m2.toml` | 0 → 500k | Half BT4, half game | 5e-4 after warm-up, then ×0.316 at 400k and ×0.1 at 466k | ~23 h |
| B | `configs/m2/m2_stage2.toml` | 500k → 750k | BT4 only | Back up to 1.58e-4, then 5e-5 at 625k, 1.58e-5 at 700k | ~11 h |
| C | `configs/m2/m2_stage3.toml` | 750k → 900k | BT4 only | Back up to 1.58e-4, then 5e-5 at 825k, 1.58e-5 at 875k | ~6.5 h |

```powershell
.venv\Scripts\python.exe -I -m mouse_moe.train --config configs\m2\m2.toml
.venv\Scripts\python.exe -I -m mouse_moe.train --config configs\m2\m2_stage2.toml --fork-from runs\m2\checkpoints\step_0500000.pt
.venv\Scripts\python.exe -I -m mouse_moe.train --config configs\m2\m2_stage3.toml --fork-from runs\m2-stage2\checkpoints\step_0750000.pt
```

- **Batch and optimiser:** 1,024 positions per step, NAdam (betas 0.9 and 0.98), weight
  decay 1e-4, gradient clip 10, bfloat16, `torch.compile`.
- **Weight averaging (SWA):** every 100 steps from 250k in stage A, 625k in stage B and
  825k in stage C. Each stage's net is its average.
- **Forking:** `--fork-from` continues another run's raw weights and optimiser state as a
  new run, with the new config's schedule.
- **Resuming:** add `--resume` to the same command. It refuses if the config has changed.
  The model, optimiser, schedule and averages come back exactly; the data order is
  reshuffled, not replayed.
- **Memory limits:** training caps its host memory so at least 4 GiB of RAM stays free
  (`--reserve-ram-gib`), and caps the GPU allocator below the card's memory, so a run
  fails with an out-of-memory error instead of slowing the machine down.
- **Progress:** each run writes JSON lines to `runs\<name>\metrics.jsonl`, with
  validation every 25k steps on 51,200 positions from the held-out test80 hour.

What healthy looks like: `expert_load_cv` below about 0.3, `grad_norm` mostly below 10, and
validation policy loss flat or falling. A non-finite gradient stops the run on purpose.

## 7. Export for the Mouse engine

```powershell
.venv\Scripts\python.exe -I -m mouse_moe.export_plugin --checkpoint runs\m2-stage3\checkpoints\step_0900000.pt --output nets\m2.1-trt.onnx --plugin-library <path to mouse_moe_trt.dll>
```

This uses the averaged weights (`--weights raw` for the live ones), builds the file in FP16
with TensorRT and the expert plugin, and compares it with PyTorch on real validation
positions. The file is deleted if they disagree.

Run it in Mouse with the UCI option `WeightsFile` set to the exported file, with
`mouse_moe_trt.dll` beside `mouse.exe`.
