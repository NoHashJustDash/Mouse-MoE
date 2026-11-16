# SPDX-License-Identifier: GPL-3.0-or-later
"""Structural validation on the downloaded Stage A shards (marker: data)."""

import pytest

from conftest import DATA_DIR
from mouse_moe.shards import iter_games, load_manifest, verified_shards
from mouse_moe.validate_data import Report, validate_records

pytestmark = pytest.mark.data


def test_stage_a_manifest_and_geometry() -> None:
    if not (DATA_DIR / "MANIFEST.csv").is_file():
        pytest.fail(f"test80 data not found at {DATA_DIR}; run without -m data or download it")
    names = sorted(load_manifest(DATA_DIR))
    assert len(names) == 24
    report = Report()
    for path in verified_shards(DATA_DIR, names):
        for count, (member, raw) in enumerate(iter_games(path)):
            if count >= 10:
                break
            validate_records(raw, report, f"{path.name}:{member}")
    assert report.games == 240
    assert not report.failures, report.examples
    assert report.castling_encoding.get("king_two_squares", 0) == 0


def test_mirrored_real_positions_keep_every_legal_move_on_an_own_piece() -> None:
    # Ties the plane mirror to the policy permutation on real games: in every
    # castling-free position, each legal move must start on a square holding one of the
    # side to move's pieces (planes 0..5 of the newest history step), mirrored or not.
    import numpy as np

    from mouse_moe.augment import mirror_files, mirrorable
    from mouse_moe.policy_map import decode
    from mouse_moe.v6 import concat_batches, expand_planes, parse_records, records_to_batch

    shard = sorted(DATA_DIR.glob("*.tar"))[0]
    games = []
    for count, (_, raw) in enumerate(iter_games(shard)):
        if count >= 20:
            break
        games.append(records_to_batch(parse_records(raw)))
    batch = concat_batches(games)
    rows = mirrorable(batch)
    assert rows.sum() > 100  # endgames and castled middlegames exist in 20 games
    sources = np.array([decode(i)[0] for i in range(1858)])
    for current in (batch, mirror_files(batch, rows)):
        own = expand_planes(current.packed, current.aux)[:, 0:6].reshape(len(current), 6, 64)
        occupied = own.any(axis=1)
        for r in np.flatnonzero(rows):
            legal = np.flatnonzero(current.policy[r] >= 0)
            assert occupied[r, sources[legal]].all()
