# SPDX-License-Identifier: GPL-3.0-or-later
"""Left-right mirror augmentation for Lc0 V6 positions (NumPy only).

Chess is symmetric under swapping the a- and h-files once castling is impossible:
castling is the only rule that tells the king's side from the queen's. Positions where
neither side keeps a castling right can therefore be mirrored file-wise, inputs and
policy target together, and remain correct training positions. Every other plane
(history, repetitions, side to move, rule-50, the constant plane) is unchanged by a
file mirror. Input planes are packed MSB-first, one byte per rank (a..h), so the mirror
reverses the bits of every byte; the policy target moves to each move's mirrored twin.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import numpy.typing as npt

from mouse_moe.policy_map import POLICY_SIZE, policy_index_of, policy_moves
from mouse_moe.v6 import Batch

# REVERSE_BITS[b] is byte b with its bit order reversed (a-file <-> h-file in a rank).
REVERSE_BITS = np.array([int(f"{b:08b}"[::-1], 2) for b in range(256)], dtype=np.uint8)
CASTLING = slice(0, 4)  # aux features 0..3: our O-O-O, our O-O, their O-O-O, their O-O


def _mirror_square(name: str) -> str:
    return "abcdefgh"[7 - "abcdefgh".index(name[0])] + name[1]


@cache
def policy_mirror() -> npt.NDArray[np.intp]:
    """``permutation[i]`` is the policy index of move ``i`` with files mirrored.

    The policy geometry (queen lines, knight jumps, promotions within one file) is
    symmetric under the mirror, so this is a permutation and its own inverse.
    """
    index = policy_index_of()
    permutation = np.array(
        [
            index[_mirror_square(move[0:2]) + _mirror_square(move[2:4]) + move[4:]]
            for move in policy_moves()
        ],
        dtype=np.intp,
    )
    if len(permutation) != POLICY_SIZE or not np.array_equal(
        permutation[permutation], np.arange(POLICY_SIZE)
    ):
        raise AssertionError("the policy file mirror is not an involution")
    return permutation


def mirrorable(batch: Batch) -> npt.NDArray[np.bool_]:
    """Positions where no side keeps a castling right."""
    castling: npt.NDArray[np.bool_] = (batch.aux[:, CASTLING] != 0).any(axis=1)
    return ~castling


def mirror_files(batch: Batch, rows: npt.NDArray[np.bool_]) -> Batch:
    """``batch`` with the selected rows mirrored file-wise (inputs and policy target)."""
    if (rows & ~mirrorable(batch)).any():
        raise ValueError("cannot mirror a position with castling rights")
    if not rows.any():
        return batch
    packed = batch.packed.copy()
    packed[rows] = REVERSE_BITS[packed[rows]]
    policy = batch.policy.copy()
    # Move i of a mirrored position is move permutation[i] of the original (involution).
    policy[rows] = policy[rows][:, policy_mirror()]
    return Batch(
        packed=packed,
        aux=batch.aux,
        policy=policy,
        wdl=batch.wdl,
        q_wdl=batch.q_wdl,
        plies_left=batch.plies_left,
    )


def random_mirror(batch: Batch, rng: np.random.Generator, probability: float = 0.5) -> Batch:
    """Mirror each castling-free position independently with ``probability``."""
    return mirror_files(batch, mirrorable(batch) & (rng.random(len(batch)) < probability))
