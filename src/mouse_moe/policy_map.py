# SPDX-License-Identifier: GPL-3.0-or-later
"""Lc0's 1858-entry policy ordering and the attention-head gather tables.

Ordering (matches lc0 v0.32.1 ``neural/encoder.cc`` and the Mouse engine's
``policy_map.cpp``): 1792 ordinary moves, enumerated by source square a1..h8 and
then target square a1..h8, keeping queen-line and knight geometry; then 66
promotions (rank 7 -> rank 8, file distance <= 1) for pieces q, r, b in that
order. Knight promotion is the plain 4-character move. Everything is from the
side to move's perspective (black positions are rank-mirrored upstream).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import numpy.typing as npt

POLICY_SIZE = 1858
ORDINARY_SIZE = 1792
PROMOTION_PIECES = "qrb"
# Promotion-bias channels in the attention head: q, r, b, then knight for the
# plain rank-7 -> rank-8 move.
PROMOTION_CHANNELS = 4
KNIGHT_CHANNEL = 3


def square_name(square: int) -> str:
    return "abcdefgh"[square % 8] + str(square // 8 + 1)


def square_index(name: str) -> int:
    if len(name) != 2 or name[0] not in "abcdefgh" or name[1] not in "12345678":
        raise ValueError(f"invalid square name: {name!r}")
    return "abcdefgh".index(name[0]) + 8 * (int(name[1]) - 1)


def _reachable(source: int, target: int) -> bool:
    dx = abs(source % 8 - target % 8)
    dy = abs(source // 8 - target // 8)
    if source == target:
        return False
    return dx == 0 or dy == 0 or dx == dy or {dx, dy} == {1, 2}


@cache
def policy_moves() -> tuple[str, ...]:
    ordinary = [
        square_name(s) + square_name(t) for s in range(64) for t in range(64) if _reachable(s, t)
    ]
    promotions = [
        square_name(s) + square_name(t) + piece
        for s in range(48, 56)
        for t in range(56, 64)
        if abs(s % 8 - t % 8) <= 1
        for piece in PROMOTION_PIECES
    ]
    moves = tuple(ordinary + promotions)
    if len(ordinary) != ORDINARY_SIZE or len(moves) != POLICY_SIZE:
        raise AssertionError("policy geometry does not produce 1792 + 66 entries")
    return moves


@cache
def policy_index_of() -> dict[str, int]:
    return {move: index for index, move in enumerate(policy_moves())}


def decode(index: int) -> tuple[int, int, str]:
    """Return (from_square, to_square, promotion) where promotion is '' or q/r/b."""
    move = policy_moves()[index]
    return square_index(move[0:2]), square_index(move[2:4]), move[4:]


def is_rank7_to_rank8(source: int, target: int) -> bool:
    return 48 <= source < 56 and 56 <= target < 64 and abs(source % 8 - target % 8) <= 1


@cache
def attention_gather_tables() -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Index tables mapping the attention head onto the 1858 outputs.

    ``base[i]`` indexes the flattened 64x64 source/destination logits.
    ``promo[i]`` indexes the flattened 8x4 promotion-bias matrix (destination
    file x channel), or 32, which the head maps to a constant zero.
    """
    base = np.empty(POLICY_SIZE, dtype=np.int64)
    promo = np.full(POLICY_SIZE, 8 * PROMOTION_CHANNELS, dtype=np.int64)
    for index in range(POLICY_SIZE):
        source, target, piece = decode(index)
        base[index] = source * 64 + target
        if piece:
            promo[index] = (target - 56) * PROMOTION_CHANNELS + PROMOTION_PIECES.index(piece)
        elif is_rank7_to_rank8(source, target):
            promo[index] = (target - 56) * PROMOTION_CHANNELS + KNIGHT_CHANNEL
    return base, promo
