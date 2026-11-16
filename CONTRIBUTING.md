# Contributing

Bug reports, fixes, ports and stronger nets are all welcome.

## Submitting a net

A net counts as stronger once it wins an SPRT. To submit one:

1. Train it and export it ([docs/6-export-and-test.md](docs/6-export-and-test.md)).
2. Test it against the current release net with the SPRT settings on that page: the same
   Mouse build on both sides, 10+0.1, bounds [0, 5].
3. Open a "Net submission" issue with:
   - the config (and the checkpoint it started from, if any);
   - the data it used, with any new files' sizes and SHA-256;
   - fastchess's final results block and your GPU and CPU;
   - a link to the exported `.onnx`.

A net that passes is re-tested on the maintainer's machine before it ships, and its
author is credited in the release notes.

Nets trained on Lc0's data must carry the notice in [docs/3-data.md](docs/3-data.md).

## Code changes

- Run `scripts\setup.ps1`: lint, type checks and tests must pass with none skipped.
- Every bug fix comes with a test that failed before it.
- Fail loudly: no silent fallbacks to another device, file or default.
- Every source file starts with `# SPDX-License-Identifier: GPL-3.0-or-later`.
- Leave the configs under `configs/` unchanged, so released nets stay reproducible. Add
  new ones.
- Say in the pull request how you verified the change, with the command and its output.

## Ports

Training only runs on Windows today ([docs/2-getting-started.md](docs/2-getting-started.md#platforms)).
A Linux port of the memory limits in `mouse_moe.resources`, with tests, is the most useful
contribution.
