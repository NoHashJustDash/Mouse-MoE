# SPDX-License-Identifier: GPL-3.0-or-later
"""Search labels: game replay, lc0 output parsing, label rows, excluded games, resume."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from conftest import STARTPOS_MOVES, make_record, pack_planes, write_manifest, write_shard
from mouse_moe.dataset import StreamSpec, stream_batches
from mouse_moe.policy_map import POLICY_SIZE, policy_index_of, square_index
from mouse_moe.search_label import (
    START,
    Board,
    Lc0Search,
    ReplayError,
    SearchError,
    SearchResult,
    board_of,
    label_game,
    label_shard,
    parse_search,
    play,
    position_label,
    replay,
)
from mouse_moe.shards import iter_games
from mouse_moe.teacher import TOP_K, ShardLabels, TeacherLabelError, write_shard_labels

ROOT = Path(__file__).resolve().parents[1]
MOVES = [policy_index_of()[move] for move in STARTPOS_MOVES]
PIECES = "PNBRQK"


def _board(pieces: dict[str, str]) -> Board:
    """A Board from {square: piece}, white upper case, black lower case."""
    sets: list[set[int]] = [set() for _ in range(12)]
    for square, piece in pieces.items():
        colour = 0 if piece.isupper() else 6
        sets[colour + PIECES.index(piece.upper())].add(square_index(square))
    return tuple(frozenset(s) for s in sets)


def _planes(board: Board, black_to_move: bool) -> npt.NDArray[np.uint8]:
    flip = 56 if black_to_move else 0
    white, black = list(board[:6]), list(board[6:])
    ours, theirs = (black, white) if black_to_move else (white, black)
    return pack_planes({p: [s ^ flip for s in squares] for p, squares in enumerate(ours + theirs)})


def _index(move: str) -> int:
    return policy_index_of()[move]


def test_board_of_reads_the_fixture_start_position_both_ways() -> None:
    assert board_of(make_record()["planes"][0], black_to_move=False) == START
    # Black to move: ranks are mirrored and black's pieces are "ours".
    assert board_of(_planes(START, black_to_move=True), black_to_move=True) == START


def test_castling_as_king_takes_rook_and_as_two_square_king_move() -> None:
    board = _board({"e1": "K", "h1": "R", "a1": "R", "e8": "k"})
    after, uci = play(board, True, _index("e1h1"))
    assert uci == "e1g1"
    assert after == _board({"g1": "K", "f1": "R", "a1": "R", "e8": "k"})
    after, uci = play(board, True, _index("e1a1"))
    assert uci == "e1c1"
    assert after == _board({"c1": "K", "d1": "R", "h1": "R", "e8": "k"})
    after, uci = play(board, True, _index("e1g1"))
    assert uci == "e1g1" and after == _board({"g1": "K", "f1": "R", "a1": "R", "e8": "k"})


def test_black_moves_are_mirrored_from_the_side_to_move_view() -> None:
    board = _board({"e1": "K", "e8": "k", "h8": "r"})
    # Black castles king side: in black's (mirrored) view the king takes its rook e1h1.
    after, uci = play(board, False, _index("e1h1"))
    assert uci == "e8g8"
    assert after == _board({"e1": "K", "g8": "k", "f8": "r"})
    after, uci = play(_board({"e1": "K", "e8": "k", "e7": "p"}), False, _index("e2e4"))
    assert uci == "e7e5" and after == _board({"e1": "K", "e8": "k", "e5": "p"})


def test_en_passant_removes_the_passed_pawn() -> None:
    board = _board({"e1": "K", "e8": "k", "e5": "P", "d5": "p"})
    after, uci = play(board, True, _index("e5d6"))
    assert uci == "e5d6" and after == _board({"e1": "K", "e8": "k", "d6": "P"})


def test_promotions_plain_move_is_a_knight() -> None:
    board = _board({"e1": "K", "e8": "k", "a7": "P", "b8": "r"})
    after, uci = play(board, True, _index("a7a8"))
    assert uci == "a7a8n" and after == _board({"e1": "K", "e8": "k", "a8": "N", "b8": "r"})
    after, uci = play(board, True, _index("a7b8q"))
    assert uci == "a7b8q" and after == _board({"e1": "K", "e8": "k", "b8": "Q"})


def test_impossible_moves_are_refused() -> None:
    with pytest.raises(ReplayError, match="no piece"):
        play(START, True, _index("e3e4"))
    with pytest.raises(ReplayError, match="own piece"):
        play(START, True, _index("d1d2"))


def _game(boards: list[Board], played: list[str], legal: list[int] = MOVES) -> npt.NDArray[np.void]:
    """Records for consecutive positions; ``played[i]`` is in record i's side-to-move view."""
    probabilities = np.full(POLICY_SIZE, -1.0, dtype=np.float32)
    probabilities[legal] = 1.0 / len(legal)
    records = []
    for ply, board in enumerate(boards):
        records.append(
            make_record(
                planes=_planes(board, black_to_move=ply % 2 == 1),
                side_to_move_or_enpassant=ply % 2,
                played_idx=_index(played[ply]),
                best_idx=legal[0],
                probabilities=probabilities,
                plies_left=float(len(boards) - ply),
            )
        )
    return np.concatenate(records)


OPENING = [
    START,
    play(START, True, _index("e2e4"))[0],
]
OPENING.append(play(OPENING[1], False, _index("e2e4"))[0])  # black's e7e5
OPENING.append(play(OPENING[2], True, _index("g1f3"))[0])
OPENING_PLAYED = ["e2e4", "e2e4", "g1f3", "b1c3"]


def test_replay_rebuilds_the_moves_and_checks_every_position() -> None:
    assert replay(_game(OPENING, OPENING_PLAYED)) == ["e2e4", "e7e5", "g1f3"]
    wrong = ["e2e4", "d2d4", "g1f3", "b1c3"]  # black really played e7e5
    with pytest.raises(ReplayError, match="ply 1"):
        replay(_game(OPENING, wrong))


def test_games_from_other_start_positions_are_not_replayed() -> None:
    elsewhere = [OPENING[1], OPENING[2]]
    assert replay(_game(elsewhere, ["e2e4", "g1f3"])) is None


# Real lc0 v0.32.1 output for 1. e4, black to move (VerboseMoveStats, BT4).
LC0_LINES = [
    "info string g7g5  (378 ) N:       0 (+ 0) (P:  0.74%) (WL:  0.00000) (D: 0.000) "
    "(M:  0.0) (Q:  0.00000) (U: 0.26458) (S: -0.05921) (V: -0.7148) ",
    "info string c7c5  (317 ) N:      15 (+ 0) (P:  9.10%) (WL: -0.05000) (D: 0.640) "
    "(M: 195.0) (Q: -0.05000) (U: 0.06302) (S:  0.05676) (V: -0.0102) ",
    "info string e7e5  (322 ) N:     386 (+ 0) (P: 67.80%) (WL: -0.00637) (D: 0.651) "
    "(M: 196.4) (Q: -0.00637) (U: 0.06302) (S:  0.05676) (V: -0.0102) ",
    "info string node  (  20) N:     408 (+ 0) (P: 88.44%) (WL: -0.01357) (D: 0.647) "
    "(M: 196.7) (Q: -0.01357) (V: -0.0150) ",
    "info depth 6 seldepth 13 time 122 nodes 408 score cp -1 wdl 171 651 178",
]


def test_parse_search_reads_moves_and_the_root() -> None:
    result = parse_search(LC0_LINES)
    assert result.indices == [378, 317, 322]
    assert result.visits == [0, 15, 386]
    assert result.root_visits == 408
    assert (result.wl, result.d, result.m) == (-0.01357, 0.647, 196.7)
    with pytest.raises(SearchError, match="no verbose"):
        parse_search(LC0_LINES[-1:])


# Real lc0 v0.32.1 output at ply 107 of test90 20251021-0717 training.193729720
# (a proven mate): the root prints its draw value as a negative zero.
LC0_NEGATIVE_ZERO = [
    "info string f6e7  (571 ) N:      70 (+ 0) (P: 13.96%) (WL:  1.00000) (D: 0.000) "
    "(M:  4.0) (Q:  1.00000) (U: 0.07955) (S:  1.08081) (V:  1.0000) (T)",
    "info string f6g5  (585 ) N:     118 (+ 0) (P: 21.76%) (WL:  1.00000) (D: 0.000) "
    "(M:  2.0) (Q:  1.00000) (U: 0.07395) (S:  1.08062) (V:  1.0000) (T)",
    "info string node  (   6) N:     510 (+ 2) (P: 100.0%) (WL:  1.00000) (D: -0.000) "
    "(M:  4.5) (Q:  1.00000) (V:  1.0000) (T)",
    "bestmove f6g5 ponder f3g3",
]


def test_parse_search_accepts_a_negative_zero_draw_value() -> None:
    result = parse_search(LC0_NEGATIVE_ZERO)
    assert result.indices == [571, 585]
    assert result.root_visits == 510
    assert (result.wl, result.d, result.m) == (1.0, 0.0, 4.5)
    legal = np.zeros(POLICY_SIZE, dtype=bool)
    legal[[571, 585]] = True
    label = position_label(result, legal)
    assert label.wdl.tolist() == [1.0, 0.0, 0.0]
    garbled = [*LC0_NEGATIVE_ZERO[:2], "info string node  (   6) N:     510 (WL: nan)"]
    with pytest.raises(SearchError, match="cannot read lc0's root statistics"):
        parse_search(garbled)


def test_position_label_keeps_the_most_visited_moves_and_the_root_wdl() -> None:
    legal = np.zeros(POLICY_SIZE, dtype=bool)
    legal[MOVES] = True
    visits = list(range(1, 21))  # move i has i + 1 visits
    label = position_label(SearchResult(MOVES, visits, 211, 0.2, 0.5, 30.0), legal)
    assert label.idx.tolist() == [MOVES[i] for i in range(19, 19 - TOP_K, -1)]
    np.testing.assert_allclose(label.prob, [v / 210 for v in range(20, 20 - TOP_K, -1)])
    np.testing.assert_allclose(label.wdl, [0.35, 0.5, 0.15], atol=1e-6)
    assert label.mlh == 30.0
    legal[MOVES[0]] = False
    with pytest.raises(SearchError, match="legal moves"):
        position_label(SearchResult(MOVES, visits, 211, 0.2, 0.5, 30.0), legal)


class FakeEngine:
    """Stands in for lc0: move i of the fixture's 20 gets i + 1 visits, plus the ply."""

    def __init__(self, log: list[str], fail_on_game: int | None = None) -> None:
        self.log = log
        self.fail_on_game = fail_on_game
        self.games = 0
        self.killed = False

    def new_game(self) -> None:
        self.games += 1
        self.log.append("new_game")
        if self.games == self.fail_on_game:
            raise SearchError("lc0 crashed")

    def search(self, moves: list[str]) -> SearchResult:
        ply = len(moves)
        return SearchResult(MOVES, [i + 1 + ply for i in range(20)], 0, 0.2, 0.5, 30.0 - ply)

    def close(self) -> None:
        self.log.append("close")

    def kill(self) -> None:
        self.killed = True


def _shard(tmp_path: Path) -> Path:
    excluded = _game([OPENING[1], OPENING[2]], ["e2e4", "g1f3"])
    games = [_game(OPENING, OPENING_PLAYED), excluded, _game(OPENING[:3], OPENING_PLAYED)]
    write_shard(tmp_path, "s.tar", games)
    write_manifest(tmp_path)
    return tmp_path / "s.tar"


def test_label_shard_labels_replayable_games_and_lists_the_rest(tmp_path: Path) -> None:
    shard = _shard(tmp_path)
    log: list[str] = []
    out = tmp_path / "labels"
    stats = label_shard(
        shard,
        lambda: FakeEngine(log),
        2,
        out,
        "cd" * 32,
        {"kind": "test"},
        save_every=1,  # type: ignore[arg-type, return-value]
    )
    assert stats["games"] == 2 and stats["excluded_games"] == 1 and stats["positions"] == 7
    labels = ShardLabels.open(out, "s.tar")
    names = [name for name, _ in iter_games(shard)]
    assert labels.excluded == {names[1]}
    start, length = labels.offsets[names[0]]
    assert length == 4
    np.testing.assert_allclose(labels.mlh.read(start, 4).ravel(), [30, 29, 28, 27])
    np.testing.assert_allclose(labels.wdl.read(start, 1), [[0.35, 0.5, 0.15]], atol=1e-3)
    assert labels.idx.read(start, 1)[0, 0] == MOVES[19]
    assert not (out / "s.tar.partial").exists()
    assert log.count("close") == 2


def test_label_shard_resumes_after_a_failure_and_kills_lc0(tmp_path: Path) -> None:
    shard = _shard(tmp_path)
    out = tmp_path / "labels"
    first_log: list[str] = []
    engines: list[FakeEngine] = []

    def failing() -> FakeEngine:
        engines.append(FakeEngine(first_log, fail_on_game=2))
        return engines[-1]

    with pytest.raises(SearchError, match="lc0 crashed"):
        label_shard(shard, failing, 1, out, "cd" * 32, {"kind": "test"}, save_every=1)  # type: ignore[arg-type]
    assert engines[0].killed
    assert not (out / "s.tar.done.json").exists()
    second_log: list[str] = []
    stats = label_shard(
        shard,
        lambda: FakeEngine(second_log),
        1,
        out,
        "cd" * 32,
        {"kind": "test"},  # type: ignore[arg-type, return-value]
    )
    # Only the game that failed is searched again; the first came from the partial save.
    assert second_log.count("new_game") == 1
    assert stats["games"] == 2 and stats["positions"] == 7


def test_a_failed_check_saves_finished_games_and_stops_lc0(tmp_path: Path) -> None:
    from mouse_moe.resources import MemoryReserveError
    from mouse_moe.search_label import PartialStore

    shard = _shard(tmp_path)
    out = tmp_path / "labels"
    engines: list[FakeEngine] = []
    calls = [0]

    def make() -> FakeEngine:
        engines.append(FakeEngine([]))
        return engines[-1]

    def check() -> None:
        calls[0] += 1
        if calls[0] == 2:  # the first check after the first finished game
            raise MemoryReserveError("free system RAM fell below the reserve")

    with pytest.raises(MemoryReserveError):
        label_shard(
            shard,
            make,  # type: ignore[arg-type]
            1,
            out,
            "cd" * 32,
            {"kind": "test"},
            save_every=100,
            check=check,
        )
    assert engines[0].killed
    assert len(PartialStore(out, "s.tar").load()) == 1  # kept, though no save was due


def test_loader_skips_excluded_games_and_only_those(synthetic_shards: Path, tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    for shard in sorted(synthetic_shards.glob("*.tar")):
        games = [(name, parse_len(raw)) for name, raw in iter_games(shard)]
        kept = games[1:]
        rows = sum(n for _, n in kept)
        idx = np.zeros((rows, TOP_K), dtype=np.uint16)
        idx[:, 0] = MOVES[0]
        prob = np.zeros((rows, TOP_K), dtype=np.float32)
        prob[:, 0] = 1.0
        arrays = {
            "idx": idx,
            "prob": prob,
            "wdl": np.tile(np.float32([0.2, 0.6, 0.2]), (rows, 1)),
            "mlh": np.zeros(rows, dtype=np.float32),
        }
        write_shard_labels(
            labels,
            shard.name,
            [n for n, _ in kept],
            [k for _, k in kept],
            arrays,
            "ab" * 32,
            excluded=[games[0][0]],
        )
    spec = StreamSpec(
        shards=tuple(sorted(synthetic_shards.glob("*.tar"))),
        batch_size=10,
        shuffle_buffer=20,
        sample_rate=1.0,
        readers=2,
        cycle=False,
        seed=3,
        teacher_labels=labels,
    )
    positions = sum(len(b) for b in stream_batches(spec))
    assert positions == 3 * 4 * 10  # the first of each shard's 5 games is skipped

    with pytest.raises(TeacherLabelError, match="both labelled and excluded"):
        write_shard_labels(labels, "x.tar", ["g"], [0], arrays, "ab" * 32, excluded=["g"])


def parse_len(raw: bytes) -> int:
    from mouse_moe.v6 import parse_records

    return len(parse_records(raw))


TEST90 = ROOT / "data" / "test90" / "training-run3-test90-20251106-0917.tar"
# lc0 and the BT4 weights file (BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz).
LC0 = Path(os.environ.get("MOUSE_MOE_LC0", ""))
BT4 = Path(os.environ.get("MOUSE_MOE_BT4", ""))


@pytest.mark.data
@pytest.mark.skipif(not TEST90.is_file(), reason="needs the downloaded test90 test shard")
def test_real_games_replay_position_by_position() -> None:
    from mouse_moe.v6 import parse_records

    replayed = excluded = 0
    for count, (_, raw) in enumerate(iter_games(TEST90)):
        if count >= 300:
            break
        moves = replay(parse_records(raw))
        replayed += moves is not None
        excluded += moves is None
    assert replayed >= 270  # about 96% of test90 games start at the initial position


@pytest.mark.data
@pytest.mark.lc0
@pytest.mark.skipif(
    not (TEST90.is_file() and LC0.is_file() and BT4.is_file()),
    reason="needs test90 data, MOUSE_MOE_LC0 and MOUSE_MOE_BT4",
)
def test_lc0_labels_a_real_game_on_its_legal_moves() -> None:
    from mouse_moe.v6 import RECORD_DTYPE, parse_records

    for _, raw in iter_games(TEST90):
        records = parse_records(raw)
        if replay(records) is not None:
            break
    short = records[:6].tobytes()
    assert len(short) == 6 * RECORD_DTYPE.itemsize
    engine = Lc0Search(LC0, BT4, "cuda-fp16", 100, 20000)
    try:
        labels = label_game("g", short, engine.new_game, engine.search)
    finally:
        engine.close()
    assert len(labels) == 6 and not labels.excluded
    sums = labels.prob.astype(np.float32).sum(axis=1)
    assert (sums > 0.9).all() and (sums <= 1.001).all()
    np.testing.assert_allclose(labels.wdl.astype(np.float32).sum(axis=1), 1.0, atol=2e-3)
