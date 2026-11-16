# SPDX-License-Identifier: GPL-3.0-or-later
"""Teacher-label storage, alignment through the loader, labeler and run forking."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import STARTPOS_MOVES
from mouse_moe.dataset import StreamSpec, build_validation_set, stream_batches
from mouse_moe.policy_map import policy_index_of
from mouse_moe.shards import iter_games
from mouse_moe.teacher import (
    TOP_K,
    ShardLabels,
    TeacherLabelError,
    require_labels,
    write_shard_labels,
)
from mouse_moe.v6 import parse_records, records_to_batch

MOVES = [policy_index_of()[move] for move in STARTPOS_MOVES]
TEACHER_SHA = "ab" * 32


def _write_labels(shards: Path, labels: Path, mlh_offset: float = 1000.0) -> None:
    """Teacher labels whose moves-left is the data's unique ply label + ``mlh_offset``.

    Row r prefers move r % 20 (0.6) over move (r + 1) % 20 (0.3); the other columns are 0.
    """
    for shard in sorted(shards.glob("*.tar")):
        names, lengths, plies = [], [], []
        for name, raw in iter_games(shard):
            game = records_to_batch(parse_records(raw))
            names.append(name)
            lengths.append(len(game))
            plies.append(game.plies_left)
        rows = int(sum(lengths))
        idx = np.zeros((rows, TOP_K), dtype=np.uint16)
        prob = np.zeros((rows, TOP_K), dtype=np.float32)
        for r in range(rows):
            idx[r, 0], idx[r, 1] = MOVES[r % 20], MOVES[(r + 1) % 20]
            prob[r, :2] = 0.6, 0.3
        wdl = np.tile(np.array([[0.25, 0.5, 0.25]], dtype=np.float32), (rows, 1))
        mlh = np.concatenate(plies) + mlh_offset
        arrays = {"idx": idx, "prob": prob, "wdl": wdl, "mlh": mlh}
        write_shard_labels(labels, shard.name, names, lengths, arrays, TEACHER_SHA)


def _spec(shards: Path, labels: Path | None, **overrides: object) -> StreamSpec:
    settings: dict[str, object] = {
        "shards": tuple(sorted(shards.glob("*.tar"))),
        "batch_size": 10,
        "shuffle_buffer": 20,
        "sample_rate": 0.5,
        "readers": 2,
        "cycle": False,
        "seed": 3,
        "teacher_labels": labels,
    }
    settings.update(overrides)
    return StreamSpec(**settings)  # type: ignore[arg-type]


def test_teacher_targets_stay_aligned_through_sampling_and_shuffling(
    synthetic_shards: Path, tmp_path: Path
) -> None:
    labels = tmp_path / "labels"
    _write_labels(synthetic_shards, labels)
    plain = list(stream_batches(_spec(synthetic_shards, None)))
    taught = list(stream_batches(_spec(synthetic_shards, labels)))
    original = np.sort(np.concatenate([b.plies_left for b in plain]))
    relabelled = np.sort(np.concatenate([b.plies_left for b in taught]))
    # Same sampled positions, each carrying its own teacher row (data label + 1000).
    np.testing.assert_array_equal(relabelled, original + 1000.0)
    for batch in taught:
        np.testing.assert_allclose(batch.wdl, [[0.25, 0.5, 0.25]] * len(batch))
        np.testing.assert_array_equal(batch.q_wdl, batch.wdl)
        # Rows restart in each shard; fixture shards hold 50 uniquely labelled positions.
        rows = (batch.plies_left - 1000.0).astype(int) % 50
        legal = batch.policy >= 0
        assert legal.sum(axis=1).tolist() == [20] * len(batch)  # illegal moves stay -1
        for i, row in enumerate(rows):
            want = np.zeros(1858, dtype=np.float32)
            want[MOVES[row % 20]], want[MOVES[(row + 1) % 20]] = 2 / 3, 1 / 3
            np.testing.assert_allclose(batch.policy[i][legal[i]], want[legal[i]], atol=1e-3)


def test_validation_set_carries_teacher_targets_on_the_same_positions(
    synthetic_shards: Path, tmp_path: Path
) -> None:
    labels = tmp_path / "labels"
    _write_labels(synthetic_shards, labels)
    shard = synthetic_shards / "shard-2.tar"
    plain = build_validation_set(shard, positions=12, per_game=3, seed=7)
    taught = build_validation_set(
        shard, positions=12, per_game=3, seed=7, teacher=ShardLabels.open(labels, shard.name)
    )
    np.testing.assert_array_equal(taught.packed, plain.packed)
    # Each position keeps its own teacher row (data label + 1000).
    np.testing.assert_array_equal(taught.plies_left, plain.plies_left + 1000.0)
    np.testing.assert_allclose(taught.wdl, [[0.25, 0.5, 0.25]] * len(taught))


def test_game_only_shards_keep_their_own_targets_beside_labelled_shards(
    synthetic_shards: Path, tmp_path: Path
) -> None:
    labels = tmp_path / "labels"
    _write_labels(synthetic_shards, labels)
    # The game-only shard has no labels at all, so opening them would fail loudly.
    (labels / "shard-1.tar.done.json").unlink()
    plain = list(stream_batches(_spec(synthetic_shards, None)))
    mixed = list(
        stream_batches(_spec(synthetic_shards, labels, game_only=frozenset({"shard-1.tar"})))
    )
    original = np.sort(np.concatenate([b.plies_left for b in plain]))
    got = np.sort(np.concatenate([b.plies_left for b in mixed]))
    # Fixture positions 50..99 are shard-1's: they keep their labels; the rest get +1000.
    in_game_shard = (original >= 50) & (original < 100)
    want = np.sort(np.where(in_game_shard, original, original + 1000.0))
    np.testing.assert_array_equal(got, want)
    assert 0 < in_game_shard.sum() < len(original)


def test_game_shards_take_their_own_label_directory_when_given(
    synthetic_shards: Path, tmp_path: Path
) -> None:
    labels, game_labels = tmp_path / "labels", tmp_path / "game-labels"
    _write_labels(synthetic_shards, labels)
    _write_labels(synthetic_shards, game_labels, mlh_offset=500.0)
    # Each set must read only its own directory, so remove the other set's labels.
    (labels / "shard-1.tar.done.json").unlink()
    for shard in synthetic_shards.glob("*.tar"):
        if shard.name != "shard-1.tar":
            (game_labels / f"{shard.name}.done.json").unlink()
    plain = list(stream_batches(_spec(synthetic_shards, None)))
    spec = _spec(
        synthetic_shards,
        labels,
        game_only=frozenset({"shard-1.tar"}),
        game_labels=game_labels,
    )
    original = np.sort(np.concatenate([b.plies_left for b in plain]))
    got = np.sort(np.concatenate([b.plies_left for b in stream_batches(spec)]))
    # Fixture positions 50..99 are shard-1's: they get +500; the rest get +1000.
    in_game_shard = (original >= 50) & (original < 100)
    want = np.sort(np.where(in_game_shard, original + 500.0, original + 1000.0))
    np.testing.assert_array_equal(got, want)
    assert 0 < in_game_shard.sum() < len(original)


def test_missing_or_mismatched_labels_fail_loudly(synthetic_shards: Path, tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    shards = sorted(synthetic_shards.glob("*.tar"))
    with pytest.raises(TeacherLabelError, match="labels missing"):
        require_labels(labels, shards)
    with pytest.raises(TeacherLabelError, match="no complete teacher labels"):
        ShardLabels.open(labels, shards[0].name)
    _write_labels(synthetic_shards, labels)
    require_labels(labels, shards)
    opened = ShardLabels.open(labels, shards[0].name)
    name, raw = next(iter_games(shards[0]))
    game = records_to_batch(parse_records(raw))
    with pytest.raises(TeacherLabelError, match="labelled plies"):
        opened.apply(replace(game, policy=game.policy[:-1], packed=game.packed[:-1]), name)
    with pytest.raises(TeacherLabelError, match="no labels for game"):
        opened.apply(game, "games/unknown.gz")
    illegal = game.policy.copy()
    illegal[0, MOVES[0]] = -1.0
    with pytest.raises(TeacherLabelError, match="illegal move"):
        opened.apply(replace(game, policy=illegal), name)


def test_labels_are_read_from_disk_not_memory_mapped(
    synthetic_shards: Path, tmp_path: Path
) -> None:
    # Regression: memory-mapped labels stayed in the loader workers' working sets and
    # sank available RAM to the training memory guard's reserve (M1 stage 2, 2026-10-26).
    labels = tmp_path / "labels"
    _write_labels(synthetic_shards, labels)
    shard = sorted(synthetic_shards.glob("*.tar"))[0].name
    opened = ShardLabels.open(labels, shard)
    assert not any(isinstance(v, np.memmap) for v in vars(opened).values())
    for key in ("idx", "prob", "wdl", "mlh"):
        whole = np.load(labels / f"{shard}.{key}.npy")
        rows = getattr(opened, key)
        np.testing.assert_array_equal(rows.read(7, 13), whole[7:20])
        assert rows.read(0, 0).shape == (0, *whole.shape[1:])
    with pytest.raises(TeacherLabelError, match="rows 45:55 of 50"):
        opened.mlh.read(45, 10)
    path = labels / f"{shard}.mlh.npy"
    path.write_bytes(path.read_bytes()[:-4])  # cut off the last two float16 rows
    with pytest.raises(TeacherLabelError, match="file ends before row 50"):
        ShardLabels.open(labels, shard).mlh.read(40, 10)


def test_an_interrupted_write_leaves_no_done_file(synthetic_shards: Path, tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    _write_labels(synthetic_shards, labels)
    shard = sorted(synthetic_shards.glob("*.tar"))[0].name
    bad = {
        "idx": np.zeros((3, TOP_K)),
        "prob": np.zeros((3, TOP_K)),
        "wdl": np.zeros((3, 3)),
        "mlh": np.zeros(2),
    }
    with pytest.raises(TeacherLabelError, match="mlh has shape"):
        write_shard_labels(labels, shard, ["g"], [3], bad, TEACHER_SHA)
    assert (labels / f"{shard}.done.json").is_file()  # shape is checked before anything changes


torch = pytest.importorskip("torch", reason="PyTorch is installed by scripts/setup.ps1")

from mouse_moe.label_teacher import label_shard, select_top_moves  # noqa: E402
from mouse_moe.train import _fork_identity, _resume_identity  # noqa: E402


@pytest.mark.torch
def test_select_top_moves_never_puts_mass_on_illegal_moves() -> None:
    logits = torch.tensor([[5.0, 4.0, 3.0, 9.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    legal = torch.tensor([[True, True, True, False, True], [False, True, False, False, False]])
    idx, prob = select_top_moves(logits, legal, k=3)
    assert idx[0].tolist() == [0, 1, 2]
    torch.testing.assert_close(
        prob[0].sum(), torch.softmax(torch.tensor([5.0, 4, 3, 1]), 0)[:3].sum()
    )
    assert idx[1, 0].item() == 1 and prob[1].tolist() == [1.0, 0.0, 0.0]


@pytest.mark.torch
def test_labeler_writes_aligned_labels_for_every_game(
    synthetic_shards: Path, tmp_path: Path
) -> None:
    shard = sorted(synthetic_shards.glob("*.tar"))[0]
    target = policy_index_of()["d2d4"]

    def evaluate(planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = torch.zeros(planes.shape[0], 1858)
        logits[:, target] = 5.0
        wdl = torch.tensor([[0.1, 0.8, 0.1]]).expand(planes.shape[0], 3)
        return logits, wdl, torch.full((planes.shape[0], 1), 42.0)

    stats = label_shard(shard, evaluate, torch.device("cpu"), tmp_path, TEACHER_SHA)
    assert stats["positions"] == 50 and stats["games"] == 5
    labels = ShardLabels.open(tmp_path, shard.name)
    assert list(labels.offsets) == [name for name, _ in iter_games(shard)]
    rows = len(labels.idx)
    assert (labels.idx.read(0, rows)[:, 0] == target).all()
    assert (labels.mlh.read(0, rows) == 42.0).all()
    # 20 legal moves, 16 kept: the four dropped ones each had e^0 of the e^5 + 19 total.
    assert stats["top_k_mass"] == pytest.approx((np.exp(5) + 15) / (np.exp(5) + 19), abs=1e-3)


def test_a_fork_may_change_only_the_name_and_the_targets() -> None:
    parent = {"name": "a", "model": {"d_model": 64}, "data": {"teacher_labels": "x", "workers": 2}}
    child = {"name": "b", "model": {"d_model": 64}, "data": {"teacher_labels": "", "workers": 1}}
    assert _fork_identity(parent) == _fork_identity(child)
    other_model = {**child, "model": {"d_model": 128}}
    assert _fork_identity(parent) != _fork_identity(other_model)


def test_a_fork_may_change_its_schedule_and_drop_game_shards_but_not_the_optimizer() -> None:
    parent: dict[str, Any] = {
        "name": "a",
        "model": {"d_model": 64},
        "data": {"teacher_weight": 0.5, "game_directory": "g", "game_shards": ["g1.tar"]},
        "train": {
            "steps": 500,
            "lr_boundaries": [400],
            "lr_factors": [0.1],
            "swa_start": 250,
            "betas": [0.9, 0.98],
            "batch_size": 1024,
        },
    }
    stage2: dict[str, Any] = {
        "name": "b",
        "model": {"d_model": 64},
        "data": {"teacher_weight": 1.0, "game_directory": "", "game_shards": []},
        "train": {
            "steps": 750,
            "lr_boundaries": [400, 500, 625],
            "lr_factors": [0.1, 0.3, 0.1],
            "swa_start": 625,
            "rewarm_start": 500,
            "rewarm_steps": 2,
            "rewarm_from": 0.1,
            "betas": [0.9, 0.98],
            "batch_size": 1024,
        },
    }
    assert _fork_identity(parent) == _fork_identity(stage2)
    for key, value in (("betas", [0.9, 0.99]), ("batch_size", 512)):
        changed = {**stage2, "train": {**stage2["train"], key: value}}
        assert _fork_identity(parent) != _fork_identity(changed)
    # A resume still refuses any schedule change, and old checkpoints resume unchanged.
    assert _resume_identity(parent) != _resume_identity({**parent, "train": stage2["train"]})
    defaults = {**parent["train"], "rewarm_start": 0, "rewarm_steps": 0, "rewarm_from": 1.0}
    assert _resume_identity(parent) == _resume_identity({**parent, "train": defaults})


def test_teacher_weight_mixes_every_target_linearly(synthetic_shards: Path, tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    _write_labels(synthetic_shards, labels)
    shard = sorted(synthetic_shards.glob("*.tar"))[0]
    opened = ShardLabels.open(labels, shard.name)
    name, raw = next(iter_games(shard))
    game = records_to_batch(parse_records(raw))
    teacher = opened.apply(game, name)
    own = opened.apply(game, name, weight=0.0)
    half = opened.apply(game, name, weight=0.5)
    legal = game.policy >= 0
    np.testing.assert_allclose(own.policy[legal], game.policy[legal], atol=1e-6)
    np.testing.assert_allclose(own.wdl, game.wdl, atol=1e-6)
    np.testing.assert_allclose(own.q_wdl, game.q_wdl, atol=1e-6)
    np.testing.assert_allclose(own.plies_left, game.plies_left, atol=1e-4)
    np.testing.assert_allclose(
        half.policy[legal], 0.5 * teacher.policy[legal] + 0.5 * game.policy[legal], atol=1e-6
    )
    np.testing.assert_allclose(half.wdl, 0.5 * teacher.wdl + 0.5 * game.wdl, atol=1e-6)
    np.testing.assert_allclose(half.q_wdl, 0.5 * teacher.wdl + 0.5 * game.q_wdl, atol=1e-6)
    assert (half.policy[~legal] == -1.0).all()
    np.testing.assert_allclose(np.where(legal, half.policy, 0).sum(axis=1), 1.0, atol=1e-5)


def test_teacher_temperature_softens_the_teacher_policy(
    synthetic_shards: Path, tmp_path: Path
) -> None:
    labels = tmp_path / "labels"
    _write_labels(synthetic_shards, labels)
    shard = sorted(synthetic_shards.glob("*.tar"))[0]
    opened = ShardLabels.open(labels, shard.name)
    name, raw = next(iter_games(shard))
    game = records_to_batch(parse_records(raw))
    soft = opened.apply(game, name, temperature=2.0)
    row = soft.policy[0]
    top = np.sort(row[row > 0])[::-1]
    # 0.6 and 0.3 at temperature 2 -> sqrt ratio: 0.6**0.5 / (0.6**0.5 + 0.3**0.5).
    np.testing.assert_allclose(top[0], 0.6**0.5 / (0.6**0.5 + 0.3**0.5), atol=1e-5)
    with pytest.raises(TeacherLabelError, match="weight"):
        opened.apply(game, name, weight=1.5)
    with pytest.raises(TeacherLabelError, match="weight"):
        opened.apply(game, name, temperature=0.0)


def test_runs_saved_before_the_mixing_settings_resume_and_fork_unchanged() -> None:
    old = {"name": "a", "model": {"d_model": 64}, "data": {"teacher_labels": "x", "workers": 2}}
    new = {
        "name": "a",
        "model": {"d_model": 64},
        "data": {
            "teacher_labels": "x",
            "workers": 1,
            "teacher_weight": 1.0,
            "teacher_temperature": 1.0,
            "mirror": False,
        },
    }
    assert _resume_identity(old) == _resume_identity(new)
    no_game_shards = {**new, "data": {**new["data"], "game_directory": "", "game_shards": ()}}
    assert _resume_identity(old) == _resume_identity(no_game_shards)
    mixed = {**new, "name": "b", "data": {**new["data"], "teacher_weight": 0.5, "mirror": True}}
    assert _fork_identity(old) == _fork_identity(mixed)
    assert _resume_identity(old) != _resume_identity(mixed)
