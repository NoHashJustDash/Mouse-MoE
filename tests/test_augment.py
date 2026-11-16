# SPDX-License-Identifier: GPL-3.0-or-later
"""File-mirror augmentation: exact policy twins, plane geometry, castling safety."""

from dataclasses import replace

import numpy as np
import pytest

from mouse_moe.augment import mirror_files, mirrorable, policy_mirror, random_mirror
from mouse_moe.policy_map import POLICY_SIZE, policy_index_of
from mouse_moe.v6 import AUX_FEATURES, PLANE_BYTES, Batch, expand_planes


def _batch(rows: int, castling: bool, seed: int = 0) -> Batch:
    rng = np.random.default_rng(seed)
    aux = np.zeros((rows, AUX_FEATURES), dtype=np.float32)
    aux[:, 5] = rng.integers(0, 100, rows)  # rule-50 count
    aux[:, 7] = 1.0
    if castling:
        aux[:, 1] = 1.0
    policy = np.full((rows, POLICY_SIZE), -1.0, dtype=np.float32)
    for r in range(rows):
        legal = rng.choice(POLICY_SIZE, size=20, replace=False)
        policy[r, legal] = rng.dirichlet(np.ones(20)).astype(np.float32)
    return Batch(
        packed=rng.integers(0, 256, (rows, PLANE_BYTES), dtype=np.uint8),
        aux=aux,
        policy=policy,
        wdl=rng.dirichlet(np.ones(3), rows).astype(np.float32),
        q_wdl=rng.dirichlet(np.ones(3), rows).astype(np.float32),
        plies_left=rng.integers(1, 200, rows).astype(np.float32),
    )


def test_policy_mirror_is_an_involution_with_exact_twins() -> None:
    perm = policy_mirror()
    index = policy_index_of()
    np.testing.assert_array_equal(perm[perm], np.arange(POLICY_SIZE))
    for move, twin in [("e2e4", "d2d4"), ("g1f3", "b1c3"), ("a7a8q", "h7h8q"), ("b7c8r", "g7f8r")]:
        assert perm[index[move]] == index[twin]
        assert perm[index[twin]] == index[move]


def test_mirrored_planes_are_the_file_flip_of_the_original() -> None:
    batch = _batch(32, castling=False)
    mirrored = mirror_files(batch, np.ones(32, dtype=bool))
    before = expand_planes(batch.packed, batch.aux)
    after = expand_planes(mirrored.packed, mirrored.aux)
    # [N, plane, rank, file]: a file mirror reverses the last axis, ranks untouched.
    np.testing.assert_array_equal(after, before[..., ::-1])


def test_mirrored_policy_puts_each_target_on_its_twin() -> None:
    batch = _batch(8, castling=False, seed=1)
    mirrored = mirror_files(batch, np.ones(8, dtype=bool))
    perm = policy_mirror()
    for r in range(8):
        for i in np.flatnonzero(batch.policy[r] >= 0):
            assert mirrored.policy[r, perm[i]] == batch.policy[r, i]
        assert (mirrored.policy[r] >= 0).sum() == (batch.policy[r] >= 0).sum()
    for field in ("aux", "wdl", "q_wdl", "plies_left"):
        np.testing.assert_array_equal(getattr(mirrored, field), getattr(batch, field))


def test_mirroring_twice_restores_the_batch() -> None:
    batch = _batch(16, castling=False, seed=2)
    rows = np.arange(16) % 2 == 0
    twice = mirror_files(mirror_files(batch, rows), rows)
    np.testing.assert_array_equal(twice.packed, batch.packed)
    np.testing.assert_array_equal(twice.policy, batch.policy)


def test_positions_with_castling_rights_are_never_mirrored() -> None:
    castling = _batch(16, castling=True, seed=3)
    assert not mirrorable(castling).any()
    with pytest.raises(ValueError, match="castling"):
        mirror_files(castling, np.ones(16, dtype=bool))
    unchanged = random_mirror(castling, np.random.default_rng(0), probability=1.0)
    np.testing.assert_array_equal(unchanged.packed, castling.packed)
    np.testing.assert_array_equal(unchanged.policy, castling.policy)


def test_random_mirror_flips_about_half_of_the_eligible_positions() -> None:
    free = _batch(400, castling=False, seed=4)
    mixed = replace(free, aux=free.aux.copy())
    mixed.aux[:200, 0] = 1.0  # first half can still castle queen-side
    out = random_mirror(mixed, np.random.default_rng(5))
    changed = (out.packed != mixed.packed).any(axis=1)
    assert not changed[:200].any()
    assert 70 <= changed[200:].sum() <= 130
