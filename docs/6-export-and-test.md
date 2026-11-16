# Export and test

A net is only better once it wins games. This page covers exporting a checkpoint for the
[Mouse engine](https://github.com/NoHashJustDash/Mouse), playing it, and testing it against
another net with an SPRT.

## Export

```powershell
.venv\Scripts\python.exe -I -m mouse_moe.export_plugin --checkpoint runs\<name>\checkpoints\latest.pt --output nets\<name>.onnx --plugin-library <path to mouse_moe_trt.dll>
```

- **What it writes:** the averaged (SWA) weights by default (`--weights raw` for the live
  ones), with each layer's experts as one node for Mouse's TensorRT plugin.
- **How it's checked:** the file is built in FP16 with TensorRT and compared with PyTorch
  on real validation positions. It is deleted unless at least 97% of top moves agree, the
  mean policy KL divergence is below 5e-4, and no win/draw/loss output is off by more
  than 0.15.
- **Requirements:** TensorRT and the plugin
  ([2-getting-started.md](2-getting-started.md#4-optional-tools)). The net must fit the
  plugin's limits ([5-scaling.md](5-scaling.md#limits-for-running-a-net-in-the-mouse-engine)).

`mouse_moe.export_onnx` writes a plugin-free file from standard ONNX operators, checked
the same way on ONNX Runtime's CPU provider. It works for any shape, but Mouse runs it
much more slowly.

## Play it

1. Download a Mouse release. It includes `mouse_moe_trt.dll` beside `mouse.exe`, and it
   needs CUDA 12.8, cuDNN 9.8 and TensorRT 10.9 installed (see the Mouse README).
2. Add `mouse.exe` to a chess GUI, or to a match runner, as a UCI engine.
3. Set the UCI options:
   - `WeightsFile`: your exported `.onnx`.
   - `GpuDllDirectories`: your CUDA `bin`, cuDNN `bin` and TensorRT `lib` folders,
     separated by semicolons.

The first start builds TensorRT engines for your GPU, which takes a few minutes. Later
starts reuse them.

Mouse's search values were tuned for M1.5's shape. A very different net may want
different values, so compare like with like.

## Test it with an SPRT

An SPRT (sequential probability ratio test) plays game pairs until it is confident
either way. Mouse's nets were tested with [fastchess](https://github.com/Disservin/fastchess):

```powershell
fastchess.exe `
  -engine name=new cmd=mouse.exe option.WeightsFile=nets\new.onnx "option.GpuDllDirectories=<CUDA bin>;<cuDNN bin>;<TensorRT lib>" `
  -engine name=old cmd=mouse.exe option.WeightsFile=nets\old.onnx "option.GpuDllDirectories=<CUDA bin>;<cuDNN bin>;<TensorRT lib>" `
  -each tc=10+0.1 proto=uci `
  -openings file=UHO_Lichess_4852_v1.epd format=epd order=random `
  -rounds 20000 -games 2 -repeat -concurrency 1 `
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 model=logistic `
  -draw movenumber=40 movecount=8 score=10 -resign movecount=4 score=1000 twosided=true `
  -recover -report penta=true -pgnout file=games.pgn
```

**How to read it.**
- Every 10 games fastchess prints the Elo difference with its 95% error, the game pairs
  (`Ptnml`), and the LLR.
- The test ends when the LLR reaches +2.94 (H1: the new net is better) or −2.94 (H0: it
  is not at least 5 Elo better).
- Early estimates swing widely. Wait for the decision.

**Keep the test fair.**
- Use the same engine build and settings on both sides, so only the net differs.
- Play one game at a time (`-concurrency 1`). Both engines share the GPU, and two games
  at once would change their speed.
- Keep the machine otherwise idle. A busy GPU or CPU costs nodes, and so Elo.
- Use bounds [0, 5] to show a gain, and [−5, 0] to show that a faster or smaller net loses
  nothing.
- Openings from Stefan Pohl's UHO book give decisive games, so tests finish sooner. Each
  opening is played once with each colour (`-repeat`).

The matches behind Mouse's nets, with their full results, are in
[7-results.md](7-results.md).
