# Training

How a run works, how M2.1 was trained, and how to build on it. Set up first
([2-getting-started.md](2-getting-started.md)) and get the data and labels
([3-data.md](3-data.md)).

## Running a config

```powershell
.venv\Scripts\python.exe -I -m mouse_moe.train --config examples\default.toml
```

A run writes everything to `runs\<name>\`, where `<name>` is the config's `name`:

- `config.json`: the config as it was loaded.
- `metrics.jsonl`: one JSON line per event. `train` lines carry the losses, `policy_top1`,
  `grad_norm`, `expert_load_cv` and `positions_per_s`; `validation` lines carry the same
  measures on 51,200 positions from the validation hour.
- `checkpoints\step_<n>.pt` every `checkpoint_every` steps, and `checkpoints\latest.pt`.

**Signs of a healthy run:**
- `expert_load_cv` below about 0.3, meaning the experts are used evenly.
- `grad_norm` mostly below the clip of 10.
- Validation policy loss flat or falling.

A non-finite gradient stops the run on purpose. Resume it with a lower `lr`.

**Measure speed before a long run.** The GPU should be otherwise idle:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\benchmark.ps1 -Config examples\default.toml
```

## Continuing, forking and fine-tuning

| Flag | Starts from | Keeps | Use it to |
|---|---|---|---|
| `--resume` | `runs\<name>\checkpoints\latest.pt` | everything; refuses if the config changed | carry on after a stop |
| `--fork-from <checkpoint>` | another run's checkpoint | weights and optimiser state, as a new run with this config's targets and schedule | add a stage, as M2.1 did |
| `--init-from <checkpoint>` | a checkpoint's averaged weights | weights only; optimiser, schedule and step start fresh | fine-tune a released net on new data |

All three need the same model shape as the checkpoint.

- **Resuming** restores the model, optimiser, schedule and weight averages exactly. The
  data order is reshuffled, not replayed.
- **`--init-from`** with M2.1's checkpoint is the cheapest way to try new data or targets:
  you start from a strong net instead of from zero. The checkpoint will be published with
  Mouse 2.0.0, which is not released yet.
- **Checkpoints are Python pickles,** and loading one can run code. Only load checkpoints
  from sources you trust, and check a release checkpoint's SHA-256 against its release
  notes.

## How M2.1 was trained

Three stages, each one a fork of the last:

| Stage | Config | Steps | Targets | Learning rate | Time on an RTX 5070 |
|---|---|---|---|---|---|
| A | `configs/m2/m2.toml` (= `examples/default.toml`) | 0 → 500k | half BT4, half game | 5e-4 after warm-up; ×0.316 at 400k, ×0.1 at 466k | ~23 h |
| B | `configs/m2/m2_stage2.toml` | 500k → 750k | BT4 only | back up to 1.58e-4; 5e-5 at 625k, 1.58e-5 at 700k | ~11 h |
| C | `configs/m2/m2_stage3.toml` | 750k → 900k | BT4 only | back up to 1.58e-4; 5e-5 at 825k, 1.58e-5 at 875k | ~6.5 h |

```powershell
.venv\Scripts\python.exe -I -m mouse_moe.train --config configs\m2\m2.toml
.venv\Scripts\python.exe -I -m mouse_moe.train --config configs\m2\m2_stage2.toml --fork-from runs\m2\checkpoints\step_0500000.pt
.venv\Scripts\python.exe -I -m mouse_moe.train --config configs\m2\m2_stage3.toml --fork-from runs\m2-stage2\checkpoints\step_0750000.pt
```

- **Optimiser:** 1,024 positions per step, NAdam (betas 0.9 and 0.98), weight decay 1e-4,
  gradient clip 10, bfloat16, `torch.compile`.
- **Weight averaging (SWA):** every 100 steps, from 250k in stage A, 625k in stage B and
  825k in stage C. Each stage's net is its average.
- **Routing balance:** at the end of each stage the averaged net's expert-balancing biases
  are re-tuned on 300 batches with the weights frozen (`balance_calibration_batches`).
- **Losses:** policy 1.0, win/draw/loss 1.0, the search's own win/draw/loss 0.5,
  moves-left 0.1, plus the small balance and router losses (`[loss]` and `[model.moe]`).
- **Memory limits:** a run caps its host memory so at least 4 GiB of RAM stays free
  (`--reserve-ram-gib`), and caps the GPU allocator at 92% of the card
  (`--gpu-memory-fraction`). A run that doesn't fit stops with an out-of-memory error and
  doesn't slow the machine down.

## What M1 to M2.1 showed

- **The teacher labels mattered most.** M2.0 changed only the data from M1.5: BT4 labels
  on every hour instead of game targets on half of them. That was worth +27.2 ± 12.4 Elo.
- **Validation loss is a poor guide to strength.** Stage C ended with validation loss
  slightly above stage B's (3.1127 against 3.1086), yet its net, M2.1, beat stage B's by
  +31.9 ± 13.4 Elo. Judge every net by games
  ([6-export-and-test.md](6-export-and-test.md)), and see [7-results.md](7-results.md).
- **Runs are not bit-for-bit repeatable:** the data order is reshuffled on resume. Compare
  nets in games, not by a few thousandths of loss.
