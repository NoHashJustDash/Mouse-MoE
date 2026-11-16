# SPDX-License-Identifier: GPL-3.0-or-later
"""Label training positions with lc0 searches by a teacher network (Lc0's BT4).

    python -I -m mouse_moe.search_label --lc0 <lc0.exe> `
        --weights <BT4-...-332.pb.gz> `
        --data data\\test90 --out data\\test90\\labels\\bt4-it332-search400 `
        --shards training-run3-test90-20251021-0717.tar --workers 6

``mouse_moe.label_teacher`` stores the teacher's raw network output. This module stores
what the teacher's *search* concludes instead, in the same label format
(``mouse_moe.teacher``), so training needs no other change.

**Replay.** Each game that starts at the initial position, white to move, is replayed
move by move. Every move is rebuilt from the record's ``played_idx`` and checked
against the next record's board; any mismatch stops labelling. Games that start
anywhere else cannot be replayed with their history and are listed as excluded.

**Search.** lc0 searches every position in game order (``position startpos moves ...``
then ``go nodes N``), so each search reuses the previous position's tree, as Lc0's own
training games do. ``go nodes`` counts the reused visits, so the root ends with at
least N visits (Lc0's test90 records show 400-431 for N = 400). Smart pruning is off,
so a search never stops early.

**Labels**, from lc0's verbose move statistics (lc0 v0.32.1
``src/search/classic/search.cc``, ``Search::GetVerboseStats``):

- policy: each root move's share of the visits to the root's children, keeping the
  16 most visited (``teacher.TOP_K``);
- WDL: the root's W = (1 + WL - D) / 2, D, L = (1 - WL - D) / 2, for the side to move;
- moves-left: the root's M.

lc0 must have searched exactly the record's legal moves (by policy index), or
labelling stops: that proves both saw the same position.

Partial results are saved every ``--save-every`` games, so a stopped run resumes
where it left off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import tarfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mouse_moe.policy_map import decode, square_name
from mouse_moe.resources import GIB, MemoryReserveError, available_ram, no_idle_sleep
from mouse_moe.shards import ShardError, iter_games, sha256_file, verified_shards
from mouse_moe.teacher import TOP_K, write_shard_labels
from mouse_moe.v6 import parse_records, piece_bitboards, records_to_batch

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
PROMOTION_LETTER = {"q": QUEEN, "r": ROOK, "b": BISHOP, "n": KNIGHT}

# 12 occupancy sets, absolute squares (a1 = 0): white P N B R Q K, then black P N B R Q K.
Board = tuple[frozenset[int], ...]

START: Board = (
    frozenset(range(8, 16)),
    frozenset({1, 6}),
    frozenset({2, 5}),
    frozenset({0, 7}),
    frozenset({3}),
    frozenset({4}),
    frozenset(range(48, 56)),
    frozenset({57, 62}),
    frozenset({58, 61}),
    frozenset({56, 63}),
    frozenset({59}),
    frozenset({60}),
)


class ReplayError(ValueError):
    """A record's played move does not lead to the next record's position."""


class SearchError(RuntimeError):
    """lc0 failed, or its search does not match the record it should label."""


def board_of(packed_row: npt.NDArray[np.uint8], black_to_move: bool) -> Board:
    """A record's current position (planes 0-11) as an absolute ``Board``.

    Lc0 planes are from the side to move's view, ranks mirrored when black moves.
    """
    planes = piece_bitboards(packed_row)[:12]
    flip = 56 if black_to_move else 0
    sets = [frozenset(int(square) ^ flip for square in np.flatnonzero(plane)) for plane in planes]
    ours, theirs = sets[:6], sets[6:]
    return tuple(theirs + ours) if black_to_move else tuple(ours + theirs)


def play(board: Board, white: bool, index: int) -> tuple[Board, str]:
    """Play the move with policy index ``index`` (side-to-move view); return the board and UCI.

    Lc0 encodes castling as the king taking its own rook; a two-square king move is
    accepted too. Knight promotion is the plain rank-7 to rank-8 move.
    """
    source, target, promotion = decode(index)
    if not white:
        source, target = source ^ 56, target ^ 56
    own, other = (0, 6) if white else (6, 0)
    sets = [set(squares) for squares in board]
    movers = [piece for piece in range(6) if source in sets[own + piece]]
    if len(movers) != 1:
        raise ReplayError(f"no piece of the side to move on {square_name(source)}")
    piece = movers[0]
    rank = source // 8
    castles = piece == KING and (
        target in sets[own + ROOK] or (target // 8 == rank and abs(target % 8 - source % 8) == 2)
    )
    if castles:
        kingside = target % 8 > source % 8
        king_to = rank * 8 + (6 if kingside else 2)
        rook_from = target if target in sets[own + ROOK] else rank * 8 + (7 if kingside else 0)
        rook_to = rank * 8 + (5 if kingside else 3)
        if rook_from not in sets[own + ROOK]:
            raise ReplayError(f"castling without a rook on {square_name(rook_from)}")
        sets[own + KING] = {king_to}
        sets[own + ROOK] = (sets[own + ROOK] - {rook_from}) | {rook_to}
        uci = square_name(source) + square_name(king_to)
    else:
        if any(target in sets[own + p] for p in range(6)):
            raise ReplayError(f"move onto own piece on {square_name(target)}")
        captured = [p for p in range(6) if target in sets[other + p]]
        for p in captured:
            sets[other + p].discard(target)
        if piece == PAWN and source % 8 != target % 8 and not captured:
            behind = target - 8 if white else target + 8  # en passant
            if behind not in sets[other + PAWN]:
                raise ReplayError(f"pawn capture onto empty {square_name(target)}")
            sets[other + PAWN].discard(behind)
        sets[own + piece].discard(source)
        last_rank = 7 if white else 0
        letter = ""
        if piece == PAWN and target // 8 == last_rank:
            letter = promotion or "n"
        elif promotion:
            raise ReplayError(f"promotion index {index} for a non-promoting move")
        sets[own + (PROMOTION_LETTER[letter] if letter else piece)].add(target)
        uci = square_name(source) + square_name(target) + letter
    return tuple(frozenset(s) for s in sets), uci


def replay(records: npt.NDArray[np.void], start: Board = START) -> list[str] | None:
    """UCI moves such that ``moves[:i]`` from ``start`` reach record ``i``.

    None if the game does not begin at ``start`` with white to move, the move count at
    zero and every castling right. Raises ``ReplayError`` if a played move does not
    lead to the next record's position.
    """
    first = records[0]
    if (
        int(first["side_to_move_or_enpassant"]) != 0
        or int(first["rule50_count"]) != 0
        or not all(
            int(first[f"castling_{side}_{wing}"]) == 1
            for side in ("us", "them")
            for wing in ("oo", "ooo")
        )
        or board_of(first["planes"], black_to_move=False) != start
    ):
        return None
    board = start
    moves: list[str] = []
    for ply in range(len(records) - 1):
        white = ply % 2 == 0
        board, uci = play(board, white, int(records[ply]["played_idx"]))
        following = records[ply + 1]
        if int(following["side_to_move_or_enpassant"]) != (1 if white else 0):
            raise ReplayError(f"ply {ply + 1}: side to move does not alternate")
        if board_of(following["planes"], black_to_move=white) != board:
            raise ReplayError(f"ply {ply}: {uci} does not reach the next record's position")
        moves.append(uci)
    return moves


_MOVE_LINE = re.compile(r"^info string (\S+)\s+\(\s*(\d+)\s*\)\s+N:\s+(\d+)\s")
_NODE_LINE = re.compile(
    r"^info string node\s+\(\s*\d+\s*\)\s+N:\s+(\d+)\s.*?"
    # lc0 prints rounded floats, so a zero D or M can appear as "-0.000".
    r"\(WL:\s*(-?\d+\.\d+)\)\s+\(D:\s*(-?\d+\.\d+)\)\s+\(M:\s*(-?\d+\.\d+)\)"
)


@dataclass
class SearchResult:
    indices: list[int]  # root moves, Lc0 policy index (side-to-move view)
    visits: list[int]
    root_visits: int
    wl: float
    d: float
    m: float


def parse_search(lines: list[str]) -> SearchResult:
    """lc0's verbose move statistics for the root (one line per move, then ``node``)."""
    indices: list[int] = []
    visits: list[int] = []
    root: re.Match[str] | None = None
    for line in lines:
        if line.startswith("info string node "):
            root = _NODE_LINE.match(line)
            if root is None:
                raise SearchError(f"cannot read lc0's root statistics: {line!r}")
            continue
        move = _MOVE_LINE.match(line)
        if move:
            indices.append(int(move.group(2)))
            visits.append(int(move.group(3)))
    if root is None or not indices:
        raise SearchError("lc0 printed no verbose move statistics")
    return SearchResult(
        indices,
        visits,
        int(root.group(1)),
        float(root.group(2)),
        float(root.group(3)),
        float(root.group(4)),
    )


@dataclass
class PositionLabel:
    idx: npt.NDArray[np.uint16]  # [TOP_K]
    prob: npt.NDArray[np.float32]  # [TOP_K]
    wdl: npt.NDArray[np.float32]  # [3]
    mlh: float


def position_label(result: SearchResult, legal: npt.NDArray[np.bool_]) -> PositionLabel:
    """The search's label for one position whose legal moves are ``legal`` [1858]."""
    indices = np.asarray(result.indices, dtype=np.int64)
    visits = np.asarray(result.visits, dtype=np.float64)
    expected = set(np.flatnonzero(legal).tolist())
    if len(indices) != len(expected) or set(indices.tolist()) != expected:
        raise SearchError(
            f"lc0 searched {len(indices)} moves, the record has {len(expected)} legal moves"
        )
    total = visits.sum()
    if total <= 0:
        raise SearchError("the search visited no root move")
    order = np.argsort(-visits, kind="stable")[:TOP_K]
    idx = np.zeros(TOP_K, dtype=np.uint16)
    prob = np.zeros(TOP_K, dtype=np.float32)
    idx[: len(order)] = indices[order]
    prob[: len(order)] = visits[order] / total
    win = (1.0 + result.wl - result.d) / 2.0
    loss = (1.0 - result.wl - result.d) / 2.0
    wdl = np.clip(np.array([win, result.d, loss], dtype=np.float64), 0.0, 1.0)
    return PositionLabel(idx, prob, (wdl / wdl.sum()).astype(np.float32), result.m)


class Lc0Search:
    """One lc0 process in UCI mode, searching positions of one game after another."""

    def __init__(self, lc0: Path, weights: Path, backend: str, visits: int, nncache: int) -> None:
        self.visits = visits
        self.process = subprocess.Popen(
            [
                str(lc0),
                f"--weights={weights}",
                f"--backend={backend}",
                "--verbose-move-stats",
                "--smart-pruning-factor=0",
                f"--nncache={nncache}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise SearchError("lc0 started without pipes")
        self._stdin = self.process.stdin
        self._stdout = self.process.stdout
        self._send("uci")
        self._read_until("uciok")
        self._ready()

    def _send(self, line: str) -> None:
        try:
            self._stdin.write(line + "\n")
            self._stdin.flush()
        except OSError as error:
            raise SearchError(f"lc0 stopped accepting input: {error}") from error

    def _read_until(self, prefix: str) -> list[str]:
        lines: list[str] = []
        while True:
            line = self._stdout.readline()
            if not line:
                raise SearchError(f"lc0 exited (code {self.process.poll()}) before '{prefix}'")
            line = line.rstrip()
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def _ready(self) -> None:
        self._send("isready")
        self._read_until("readyok")

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._ready()

    def search(self, moves: list[str]) -> SearchResult:
        self._send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
        self._send(f"go nodes {self.visits}")
        return parse_search(self._read_until("bestmove"))

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._send("quit")
                self.process.wait(timeout=10)
            except (SearchError, subprocess.TimeoutExpired):
                self.process.kill()

    def kill(self) -> None:
        """Stop the process at once (it may be mid-search on another thread)."""
        if self.process.poll() is None:
            self.process.kill()


Searcher = Callable[[list[str]], SearchResult]


@dataclass
class GameLabels:
    name: str
    excluded: bool
    idx: npt.NDArray[np.uint16] = field(default_factory=lambda: np.zeros((0, TOP_K), np.uint16))
    prob: npt.NDArray[np.float16] = field(default_factory=lambda: np.zeros((0, TOP_K), np.float16))
    wdl: npt.NDArray[np.float16] = field(default_factory=lambda: np.zeros((0, 3), np.float16))
    mlh: npt.NDArray[np.float16] = field(default_factory=lambda: np.zeros(0, np.float16))
    agree: int = 0  # positions whose most-visited move is the record's own best move

    def __len__(self) -> int:
        return len(self.mlh)


def label_game(name: str, raw: bytes, new_game: Callable[[], None], search: Searcher) -> GameLabels:
    """Search every position of one game, or mark it excluded if it cannot be replayed."""
    records = parse_records(raw)
    try:
        moves = replay(records)
    except ReplayError as error:
        raise ReplayError(f"{name}: {error}") from error
    if moves is None:
        return GameLabels(name, excluded=True)
    legal = records_to_batch(records).policy >= 0
    best = records["best_idx"].astype(np.int64)
    new_game()
    labels = []
    for ply in range(len(records)):
        try:
            labels.append(position_label(search(moves[:ply]), legal[ply]))
        except SearchError as error:
            raise SearchError(f"{name} ply {ply}: {error}") from error
    return GameLabels(
        name,
        excluded=False,
        idx=np.stack([x.idx for x in labels]),
        prob=np.stack([x.prob for x in labels]).astype(np.float16),
        wdl=np.stack([x.wdl for x in labels]).astype(np.float16),
        mlh=np.array([x.mlh for x in labels], dtype=np.float16),
        agree=int(sum(int(x.idx[0]) == int(b) for x, b in zip(labels, best, strict=True))),
    )


def game_names(shard: Path) -> list[str]:
    """Game member names in archive order, from tar headers only (as ``iter_games``)."""
    try:
        with tarfile.open(shard, mode="r:") as archive:
            return [m.name for m in archive.getmembers() if m.isfile() and m.name.endswith(".gz")]
    except tarfile.TarError as error:
        raise ShardError(f"{shard}: unreadable tar archive") from error


class PartialStore:
    """Labelled games saved in chunks under ``<out>/<shard>.partial`` for resuming."""

    def __init__(self, out_dir: Path, shard: str) -> None:
        self.directory = out_dir / f"{shard}.partial"
        self.directory.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, GameLabels]:
        games: dict[str, GameLabels] = {}
        for meta_path in sorted(self.directory.glob("chunk_*.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            arrays = np.load(meta_path.with_suffix(".npz"))
            offsets = np.concatenate([[0], np.cumsum(meta["lengths"])])
            for i, name in enumerate(meta["names"]):
                a, b = int(offsets[i]), int(offsets[i + 1])
                games[name] = GameLabels(
                    name,
                    False,
                    arrays["idx"][a:b],
                    arrays["prob"][a:b],
                    arrays["wdl"][a:b],
                    arrays["mlh"][a:b],
                    int(meta["agree"][i]),
                )
            for name in meta["excluded"]:
                games[name] = GameLabels(name, excluded=True)
        return games

    def save(self, chunk: list[GameLabels]) -> None:
        number = len(list(self.directory.glob("chunk_*.json")))
        stem = self.directory / f"chunk_{number:05d}"
        kept = [g for g in chunk if not g.excluded]
        empty = GameLabels("", False)
        parts = kept or [empty]
        temporary = stem.with_name(stem.name + ".tmp.npz")
        np.savez(
            temporary,
            idx=np.concatenate([g.idx for g in parts]),
            prob=np.concatenate([g.prob for g in parts]),
            wdl=np.concatenate([g.wdl for g in parts]),
            mlh=np.concatenate([g.mlh for g in parts]),
        )
        os.replace(temporary, stem.with_suffix(".npz"))
        meta = {
            "names": [g.name for g in kept],
            "lengths": [len(g) for g in kept],
            "agree": [g.agree for g in kept],
            "excluded": [g.name for g in chunk if g.excluded],
        }
        stem.with_name(stem.name + ".tmp.json").write_text(json.dumps(meta), encoding="utf-8")
        os.replace(stem.with_name(stem.name + ".tmp.json"), stem.with_suffix(".json"))

    def remove(self) -> None:
        for path in self.directory.iterdir():
            path.unlink()
        self.directory.rmdir()


SearcherFactory = Callable[[], Lc0Search]


def _work(
    jobs: queue.Queue[tuple[str, bytes] | None],
    results: queue.Queue[GameLabels | BaseException],
    make: SearcherFactory,
    stop: threading.Event,
    engines: list[Lc0Search],
) -> None:
    try:
        engine = make()
    except BaseException as error:  # reported to the main thread, never swallowed
        results.put(error)
        return
    engines.append(engine)
    try:
        while not stop.is_set():
            try:
                job = jobs.get(timeout=1.0)
            except queue.Empty:
                continue
            if job is None:
                return
            results.put(label_game(job[0], job[1], engine.new_game, engine.search))
    except BaseException as error:
        results.put(error)
    finally:
        engine.close()


def label_shard(
    shard: Path,
    make: SearcherFactory,
    workers: int,
    out_dir: Path,
    teacher_sha256: str,
    method: dict[str, object],
    save_every: int = 500,
    progress: Callable[[str], None] = print,
    progress_seconds: float = 300.0,
    check: Callable[[], None] = lambda: None,
) -> dict[str, float]:
    """Label every replayable game of ``shard`` and write its label files.

    ``check`` runs at least every few seconds and may raise to stop labelling (the
    free-RAM reserve); games labelled so far are saved and lc0 is stopped first.
    """
    store = PartialStore(out_dir, shard.name)
    done = store.load()
    names = game_names(shard)
    unknown = set(done) - set(names)
    if unknown:
        raise ShardError(f"{store.directory} holds games not in {shard.name}")
    jobs: queue.Queue[tuple[str, bytes] | None] = queue.Queue(maxsize=workers * 4)
    results: queue.Queue[GameLabels | BaseException] = queue.Queue()
    stop = threading.Event()
    engines: list[Lc0Search] = []
    wanted = {i for i, name in enumerate(names) if name not in done}
    todo = len(wanted)

    def put(job: tuple[str, bytes] | None) -> bool:
        """Queue a job unless labelling has stopped; never blocks forever."""
        while not stop.is_set():
            try:
                jobs.put(job, timeout=1.0)
                return True
            except queue.Full:
                continue
        return False

    def feed() -> None:
        try:
            for name, raw in iter_games(shard, only=wanted):
                if not put((name, raw)):
                    return
        except BaseException as error:
            results.put(error)
        finally:
            for _ in range(workers):
                put(None)

    threads = [
        threading.Thread(target=_work, args=(jobs, results, make, stop, engines), daemon=True)
        for _ in range(workers)
    ]
    feeder = threading.Thread(target=feed, daemon=True)
    for thread in threads:
        thread.start()
    feeder.start()
    started = last = time.perf_counter()
    pending: list[GameLabels] = []
    finished = positions = 0
    try:
        check()
        while finished < todo:
            try:
                item = results.get(timeout=2.0)
            except queue.Empty:
                check()
                continue
            if isinstance(item, BaseException):
                raise item
            pending.append(item)
            done[item.name] = item
            finished += 1
            positions += len(item)
            check()  # after recording the game, so a stop here keeps it
            if len(pending) >= save_every:
                store.save(pending)
                pending = []
            now = time.perf_counter()
            if now - last >= progress_seconds:
                last = now
                rate = positions / (now - started)
                left = sum(1 for n in names if n not in done)
                progress(
                    f"{shard.name}: {finished}/{todo} games, {positions} positions, "
                    f"{rate:.1f} positions/s, {left} games left"
                )
    finally:
        stop.set()
        if finished < todo:  # failed or interrupted: no lc0 may outlive the run
            for engine in engines:
                engine.kill()
        for thread in threads:
            thread.join(timeout=30)
        if pending:  # keep every finished game, also when stopping on an error
            store.save(pending)
    labelled = [done[n] for n in names if not done[n].excluded]
    excluded = [n for n in names if done[n].excluded]
    rows = sum(len(g) for g in labelled)
    write_shard_labels(
        out_dir,
        shard.name,
        [g.name for g in labelled],
        [len(g) for g in labelled],
        {
            "idx": np.concatenate([g.idx for g in labelled]),
            "prob": np.concatenate([g.prob for g in labelled]),
            "wdl": np.concatenate([g.wdl for g in labelled]),
            "mlh": np.concatenate([g.mlh for g in labelled]),
        },
        teacher_sha256,
        excluded=excluded,
        method=method,
    )
    store.remove()
    seconds = time.perf_counter() - started
    return {
        "positions": rows,
        "games": len(labelled),
        "excluded_games": len(excluded),
        "seconds_this_run": round(seconds, 1),
        "positions_per_s_this_run": round(positions / seconds, 1) if seconds else 0.0,
        "top1_matches_data": round(sum(g.agree for g in labelled) / rows, 4) if rows else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label positions with lc0 teacher searches")
    parser.add_argument("--lc0", type=Path, required=True, help="lc0 executable")
    parser.add_argument("--weights", type=Path, required=True, help="teacher network for lc0")
    parser.add_argument("--backend", default="cuda-fp16")
    parser.add_argument("--visits", type=int, default=400, help="root visits per search")
    parser.add_argument("--workers", type=int, required=True, help="lc0 processes")
    parser.add_argument("--nncache", type=int, default=200000, help="lc0 NN cache entries")
    parser.add_argument("--data", type=Path, required=True, help="shard directory")
    parser.add_argument("--shards", nargs="+", required=True, help="shard file names")
    parser.add_argument("--out", type=Path, required=True, help="label directory")
    parser.add_argument("--save-every", type=int, default=500, help="games per partial save")
    parser.add_argument(
        "--reserve-ram-gib",
        type=float,
        default=4.0,
        help="stop (resumably) if free system RAM falls below this",
    )
    args = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("run with python -I: training data is untrusted input")
    if args.visits < 1 or args.workers < 1 or args.reserve_ram_gib <= 0:
        parser.error("--visits, --workers and --reserve-ram-gib must be positive")
    reserve = int(args.reserve_ram_gib * GIB)
    if available_ram() < reserve + 2 * GIB:
        parser.error(
            f"only {available_ram() / GIB:.1f} GiB RAM free; need the "
            f"{args.reserve_ram_gib:g} GiB reserve plus 2 GiB for lc0"
        )

    def check() -> None:
        free = available_ram()
        if free < reserve:
            raise MemoryReserveError(
                f"free system RAM fell to {free / GIB:.1f} GiB, below the "
                f"{args.reserve_ram_gib:g} GiB reserve; rerun to resume"
            )

    for path in (args.lc0, args.weights):
        if not path.is_file():
            parser.error(f"{path} does not exist")
    shards = verified_shards(args.data.resolve(), args.shards)
    weights_sha256 = sha256_file(args.weights)
    method = {
        "kind": "lc0 search",
        "lc0_sha256": sha256_file(args.lc0),
        "weights": args.weights.name,
        "backend": args.backend,
        "visits": args.visits,
        "smart_pruning_factor": 0,
        "tree_reuse": "game order",
    }
    # One label set per (teacher, settings): mixing settings in a directory is refused.
    digest = hashlib.sha256(json.dumps(method, sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    settings = args.out / "SETTINGS.json"
    if settings.is_file():
        if json.loads(settings.read_text(encoding="utf-8")).get("digest") != digest:
            parser.error(f"{args.out} holds labels made with other settings")
    else:
        settings.write_text(json.dumps({"digest": digest, **method}, indent=2), encoding="utf-8")

    def make() -> Lc0Search:
        return Lc0Search(args.lc0, args.weights, args.backend, args.visits, args.nncache)

    with no_idle_sleep():
        for shard in shards:
            if (args.out / f"{shard.name}.done.json").is_file():
                print(json.dumps({"shard": shard.name, "status": "already labelled"}), flush=True)
                continue
            stats = label_shard(
                shard,
                make,
                args.workers,
                args.out,
                weights_sha256,
                method,
                save_every=args.save_every,
                progress=lambda line: print(line, flush=True),
                check=check,
            )
            print(json.dumps({"shard": shard.name, **stats}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
