# Results

All matches ran on one machine (GeForce RTX 5070, Ryzen 7 7800X3D), one game at a time, at
10 seconds plus 0.1 seconds per move, with pondering off. Openings come from
`UHO_Lichess_4852_v1.epd`, and each one is played twice, once with each colour. Margins
are 95% confidence intervals.

Every Mouse player is the [Mouse engine](https://github.com/NoHashJustDash/Mouse) at the
same build and the same search settings; only the net changes. lc0 is v0.32.1 at its
default settings.

## Mouse-MoE nets against each other

These are SPRTs with bounds [0, 5] Elo, α = β = 0.05: each test stops once it is
confident that the new net is better (H1) or that it is not at least 5 Elo better (H0).

| New net | Old net | Result | Games (+ − =) | Decision |
|---|---|---|---|---|
| M2.1 | M2.0 | +31.9 ± 13.4 Elo | 634 (+186 −128 =320) | H1, better |
| M2.0 | M1.5 | +27.2 ± 12.4 Elo | 844 (+235 −169 =440) | H1, better |

## Against lc0 with T1

lc0 played Lc0's T1-256x10-distilled-swa-2432500 net, 20.2M parameters.

| Mouse with | Result | Games (+ − =) | Test |
|---|---|---|---|
| M1.5 | −38.4 ± 15.1 Elo | 600 (+121 −187 =292) | Fixed 600 games, with the Mouse 1.1.0 release |
| M1.5 | −44.1 ± 17.1 Elo | 412 | SPRT [0, 5], H0 |
| M2.0 | −12.3 ± 9.9 Elo | 1,358 (+303 −351 =704) | SPRT [0, 5], H0 |

M2.1 has not been measured against lc0 with T1 yet. Its result will be added here when
that match is complete.

These compare engine and net together. With the same T1 net, Mouse 1.0.0 was
−59.6 ± 12.3 Elo against lc0, so part of each gap above is the engine, not the net.

## Speed

In the M2.0 match, Mouse with M2.0 searched about 50,000 positions per second and lc0 with
T1 about 44,000. M2.0 uses 8.4M parameters per position against T1's 20.2M, but the
saving shows up only partly as speed. In a profile at batch 96 the expert layers took
about a third of each encoder layer's time on the GPU.
