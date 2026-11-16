# SPDX-License-Identifier: GPL-3.0-or-later
import tarfile
from pathlib import Path

import numpy as np
import pytest

from conftest import numbered_game, write_manifest, write_shard
from mouse_moe.dataset import StreamSpec, build_validation_set, kept_plies, stream_batches
from mouse_moe.shards import ShardError, iter_games, stream_members, verified_shards


def _spec(shards: tuple[Path, ...], **overrides: object) -> StreamSpec:
    values: dict[str, object] = dict(
        shards=shards,
        batch_size=8,
        shuffle_buffer=32,
        sample_rate=1.0,
        readers=2,
        cycle=False,
        seed=3,
    )
    values.update(overrides)
    return StreamSpec(**values)  # type: ignore[arg-type]


def test_single_pass_emits_every_position_once(synthetic_shards: Path) -> None:
    shards = tuple(sorted(synthetic_shards.glob("*.tar")))
    labels = np.concatenate([b.plies_left for b in stream_batches(_spec(shards))])
    # 150 positions, batch 8 -> 18 full batches; the 6 left over are dropped.
    assert len(labels) == 144
    assert len(set(labels.tolist())) == 144
    assert set(labels.tolist()) <= set(range(150))


def test_stream_is_shuffled_and_deterministic(synthetic_shards: Path) -> None:
    shards = tuple(sorted(synthetic_shards.glob("*.tar")))
    first = np.concatenate([b.plies_left for b in stream_batches(_spec(shards))])
    second = np.concatenate([b.plies_left for b in stream_batches(_spec(shards))])
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, np.sort(first))


def test_sampling_and_cycling(synthetic_shards: Path) -> None:
    shards = tuple(sorted(synthetic_shards.glob("*.tar")))
    stream = stream_batches(_spec(shards, sample_rate=0.5, cycle=True))
    batches = [next(stream) for _ in range(40)]  # more than one pass: cycling works
    assert all(len(b) == 8 for b in batches)


def test_workers_split_every_shard_evenly_and_disjointly(synthetic_shards: Path) -> None:
    # Regression: whole shards were dealt round-robin, so 22 shards over 6 workers
    # gave some workers 4 hours and others 3 (those hours oversampled by 4/3).
    shards = tuple(sorted(synthetic_shards.glob("*.tar")))
    seen = []
    for worker in range(2):
        spec = _spec(shards, worker=worker, workers=2, batch_size=1, shuffle_buffer=16)
        seen.append(np.concatenate([b.plies_left for b in stream_batches(spec)]).astype(int))
    assert set(seen[0]).isdisjoint(seen[1])
    assert set(seen[0]) | set(seen[1]) == set(range(150))
    # 5 games per shard: worker 0 owns games 0, 2, 4 and worker 1 games 1, 3 of every shard.
    for labels, games in zip(seen, (3, 2), strict=True):
        assert np.bincount(labels // 50, minlength=3).tolist() == [games * 10] * 3


def test_sampling_covers_each_ply_once_per_period_with_both_sides_to_move() -> None:
    # Regressions: (1) an independent coin per position and pass left (1/2)^passes of
    # the data unseen (about 22% after 2.2 passes); (2) keeping every other ply gave
    # each game a single side to move per pass, and ab_m1_fixed's value loss trailed.
    for name in ("games/game_00001.gz", "other/game.gz", "x.gz"):
        passes = [kept_plies(name, 80, epoch, 2) for epoch in range(2)]
        merged = np.sort(np.concatenate(passes))
        assert merged.tolist() == list(range(80))  # each ply exactly once over 2 passes
        for kept in passes:
            assert 25 <= len(kept) <= 55  # about half per pass
            assert {0, 1} <= set((kept % 2).tolist())  # both sides to move in one pass
        assert kept_plies(name, 80, 2, 2).tolist() == passes[0].tolist()
    assert kept_plies("y.gz", 80, 0, 1).tolist() == list(range(80))


def test_two_passes_at_half_rate_cover_every_position(synthetic_shards: Path) -> None:
    shards = tuple(sorted(synthetic_shards.glob("*.tar")))
    spec = _spec(shards, sample_rate=0.5, cycle=True, batch_size=1, shuffle_buffer=8)
    stream = stream_batches(spec)
    # Two passes push 150 positions; the 8-slot buffer still holds 8 of them, so the
    # first 142 outputs all come from those two passes. Independent 50% sampling
    # would repeat about a quarter of them.
    labels = [int(next(stream).plies_left[0]) for _ in range(142)]
    assert len(set(labels)) == 142 and set(labels) <= set(range(150))


def test_skips_directory_members(synthetic_shards: Path) -> None:
    names = [name for name, _ in iter_games(next(synthetic_shards.glob("*.tar")))]
    assert len(names) == 5 and all(name.endswith(".gz") for name in names)


def test_manifest_guards(synthetic_shards: Path) -> None:
    name = "shard-0.tar"
    assert verified_shards(synthetic_shards, [name], check_hash=True)
    with pytest.raises(ShardError, match="not in"):
        verified_shards(synthetic_shards, ["missing.tar"])
    path = synthetic_shards / name
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ShardError, match="wrong size"):
        verified_shards(synthetic_shards, [name])
    write_manifest(synthetic_shards)
    data = bytearray(path.read_bytes())
    data[600] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ShardError, match="SHA-256"):
        verified_shards(synthetic_shards, [name], check_hash=True)


def test_rejects_bad_specs(synthetic_shards: Path) -> None:
    shards = tuple(sorted(synthetic_shards.glob("*.tar")))
    for bad in (
        {"sample_rate": 0.0},
        {"sample_rate": 0.3},
        {"shuffle_buffer": 4},
        {"readers": 0},
        {"worker": 2, "workers": 2},
    ):
        with pytest.raises(ValueError):
            _spec(shards, **bad)


def test_validation_set_spans_the_shard_and_each_game(synthetic_shards: Path) -> None:
    # Regression: validation used to re-read the same prefix of the held-out shard.
    shard = synthetic_shards / "shard-2.tar"  # 5 games, labels 100..149
    sample = build_validation_set(shard, positions=12, per_game=3, seed=7)
    labels = sample.plies_left.astype(int).tolist()
    assert len(labels) == 12 and len(set(labels)) == 12
    games = {label // 10 for label in labels}
    assert len(games) >= 4  # spread over the shard, not one prefix
    plies = {label % 10 for label in labels}
    assert 0 in plies and 9 in plies and plies & {4, 5}  # opening, middle and end
    again = build_validation_set(shard, positions=12, per_game=3, seed=7)
    assert again.plies_left.tolist() == sample.plies_left.tolist()


def test_validation_set_rejects_a_too_small_shard(synthetic_shards: Path) -> None:
    with pytest.raises(ValueError, match="only"):
        build_validation_set(synthetic_shards / "shard-2.tar", positions=40, per_game=3, seed=1)


def test_streaming_reads_do_not_cache_member_headers(tmp_path: Path) -> None:
    # Regression: tarfile keeps every header it streams past (~0.7 KB each, ~50 MB per
    # real shard). With 22 shards open in each of 6 workers, ab_m1_fixed ran out of
    # memory at step ~31k.
    games = [numbered_game(10 * i, 2) for i in range(40)]
    shard = write_shard(tmp_path, "many.tar", games)
    with tarfile.open(shard, mode="r|") as archive:
        seen = 0
        for _ in stream_members(archive):
            seen += 1
            assert len(archive.members) <= 1
    assert seen == 41  # the directory entry plus 40 games
    assert len(list(iter_games(shard))) == 40


def test_opening_plies_are_kept_at_the_configured_rate(synthetic_shards: Path) -> None:
    shards = tuple(sorted(synthetic_shards.glob("*.tar")))

    def labels(**overrides: object) -> np.ndarray:  # type: ignore[type-arg]
        # Batch size 1: a non-cycling stream drops only a final partial batch.
        spec = _spec(shards, sample_rate=1.0, batch_size=1, shuffle_buffer=10, **overrides)
        return np.sort(np.concatenate([b.plies_left for b in stream_batches(spec)]))

    everything = labels()
    assert everything.tolist() == list(range(150))
    # Labels are game_start + ply, ten plies a game: ply = label % 10.
    late_only = labels(opening_plies=4, opening_keep=1e-9)
    assert late_only.tolist() == [x for x in range(150) if x % 10 >= 4]
    half = labels(opening_plies=4, opening_keep=0.5)
    kept_openings = int((half % 10 < 4).sum())
    assert set(late_only.tolist()) <= set(half.tolist())
    assert 15 <= kept_openings <= 45  # 60 opening plies at rate 0.5
    with pytest.raises(ValueError, match="opening_keep"):
        _spec(shards, opening_plies=4, opening_keep=0.0)
