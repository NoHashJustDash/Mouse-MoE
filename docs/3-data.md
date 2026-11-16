# Training data

Mouse-MoE nets train on Lc0's open self-play data, with most targets replaced by the
evaluations of Lc0's BT4 network. None of the data is in this repository: you download it
yourself and the tools check it against the manifests here.

## Licence and notice

The [Leela Chess Zero training data](https://storage.lczero.org/files/training_data/) is
made available under the
[Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/), and its
individual contents under the
[Database Contents License (DbCL) 1.0](https://opendatacommons.org/licenses/dbcl/1-0/).
Every archive includes the licence as a `LICENSE` member.

A net trained on this data is a produced work under the ODbL. It needs this notice:

> Contains information from the Leela Chess Zero training data, which is made available
> under the Open Database License (ODbL).

If you publish the data itself, or a changed copy of it such as the BT4 labels, that copy
must also be under the ODbL.

## The files

| Run | Source | Hours used | Training positions |
|---|---|---:|---:|
| `test80`, 2024-04-01 | `https://storage.lczero.org/files/training_data/test80/` | 22 to train, 1 to validate, 1 held out for testing | about 182M |
| `test90`, 2025-10-17 to 2025-11-06 | `https://data.lczero.org/files/training_data/test90/` | 53 to train, 1 to validate, 1 held out for testing | about 115M |

The test90 hours are spread over the run's last three weeks, so the games come from many
of the run's nets rather than one day's. The exact files are listed in each
config's `train_shards` and `game_shards`.

## Downloading

```powershell
.venv\Scripts\python.exe -I -m mouse_moe.download --config examples\default.toml
```

This fetches every file the config trains, validates or tests on, into the config's data
folders:

- **Checked:** each file downloads as `<name>.part` and is renamed only once its size and
  SHA-256 match [`manifests/`](../manifests). A file that doesn't match is deleted and
  reported.
- **Restartable:** a stopped download resumes where it left off, and files already
  present are checked and skipped.
- **Recorded:** each data folder gets a `MANIFEST.csv`, which the training loader
  requires.
- **Safe for the disk:** it refuses to start unless 10 GiB stays free afterwards.

The whole M2.1 set is 77 files, 60 GiB. The hashes were computed from the downloaded
files; they are not checksums published by the Lc0 project.

**To use other Lc0 hours:** add their names to a config, and their sizes and SHA-256 to
`manifests/<run>/MANIFEST.csv`. The tool refuses files that aren't listed, so record
hashes you have checked yourself.

Check the records before a long run:

```powershell
.venv\Scripts\python.exe -I -m mouse_moe.validate_data --data data\test80 --games-per-shard 300
```

Every record must be Lc0's V6 format with input format 1. Every game must be a whole
number of 8,356-byte records, and each position's policy target must sum to 1 over its
legal moves.

## BT4 labels

Most training targets come from Lc0's
[BT4-1024x15x32h-swa-6147500-policytune-332](https://lczero.org/) network (about 185M
parameters), converted to ONNX with `lc0 leela2onnx`.
[`mouse_moe.label_teacher`](../src/mouse_moe/label_teacher.py) runs it once per position
in FP16 on TensorRT. No search is involved.

For each position it stores:

1. The teacher's 16 most likely legal moves and their probabilities, from a softmax over
   the legal moves only. They are not renormalised over the 16. A position with fewer than
   16 legal moves has probability 0 in the unused places.
2. The teacher's win/draw/loss and moves-left outputs.

During training each target is `teacher_weight × teacher + (1 − teacher_weight) × game`:
M2's first stage uses 0.5, its later stages 1.0 (BT4 only). The stored data never
changes.

### Making the labels (about 16 hours on an RTX 5070)

`label_teacher` labels the files in a config's `train_shards`, at about 5,000 positions
per second. It needs TensorRT and BT4 as ONNX ([2-getting-started.md](2-getting-started.md#4-optional-tools)).
M2.1's labels come from three runs into two folders:

```powershell
.venv\Scripts\python.exe -I -m mouse_moe.label_teacher --onnx bt4-it332-f16.onnx --config configs\m2\m2.toml --out data\test80\labels\bt4-it332
.venv\Scripts\python.exe -I -m mouse_moe.label_teacher --onnx bt4-it332-f16.onnx --config configs\m2\label_test90.toml --out data\test90\labels\bt4-it332
.venv\Scripts\python.exe -I -m mouse_moe.label_teacher --onnx bt4-it332-f16.onnx --config configs\m20\stage3.toml --out data\test90\labels\bt4-it332
```

The first labels the 22 test80 hours (about 10 hours), the second 48 test90 hours (about
6 hours), the third the other 5 test90 hours. A stopped run resumes at the next unfinished
file.

[`mouse_moe.search_label`](../src/mouse_moe/search_label.py) is the slower alternative:
it labels positions with real lc0 searches by BT4 instead of a single evaluation. M2.1
does not use it.

### Changes made while training

Two more steps also never change the stored data:
- **Mirroring:** positions where neither side can still castle are mirrored left to
  right half the time.
- **Fewer repeated openings:** positions from a game's first 10 plies are kept with
  probability 0.4, so openings that repeat across many games count for less.

## Teacher networks and Lc0

Lc0's networks and data are the work of the Lc0 project and the many people who
contributed games to it. Mouse-MoE is not affiliated with or endorsed by the Lc0 project.
