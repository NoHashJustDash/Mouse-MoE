# SPDX-License-Identifier: GPL-3.0-or-later
"""Shuffled, decorrelated batch streams over Lc0 training shards (NumPy only).

Positions from one game are strongly correlated, so the stream (1) reads several
shards at once and interleaves their games, (2) keeps a hashed 1/``period`` of
each game's plies, and (3) passes positions through a shuffle buffer before batching.

Every data-loader worker reads every shard but owns only the games whose index
within the shard is ``worker`` modulo ``workers``. Workers therefore get equal
shares of every hour (shard counts need not divide evenly) and no position is
duplicated across workers within a pass. Each pass keeps the next hash class, so
``period`` passes cover every position exactly once.
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mouse_moe.augment import random_mirror
from mouse_moe.shards import count_games, iter_games
from mouse_moe.teacher import ShardLabels
from mouse_moe.v6 import Batch, concat_batches, parse_records, records_to_batch, take


@dataclass(frozen=True, slots=True)
class StreamSpec:
    shards: tuple[Path, ...]
    batch_size: int
    shuffle_buffer: int
    sample_rate: float
    readers: int
    cycle: bool
    seed: int
    worker: int = 0
    workers: int = 1
    # Directory of teacher labels (``mouse_moe.teacher``); mixed into the targets if set.
    teacher_labels: Path | None = None
    teacher_weight: float = 1.0  # 1.0 replaces the game targets with the teacher's
    teacher_temperature: float = 1.0
    mirror: bool = False  # mirror castling-free positions file-wise with probability 1/2
    # Keep plies before opening_plies at rate opening_keep: self-play openings repeat
    # (test80 shard 0017: 70% of plies 0-9 duplicate another position, 0.8% of 20-39).
    opening_plies: int = 0
    opening_keep: float = 1.0
    # Shard file names that keep their games' own targets even with teacher_labels set,
    # or, with game_labels set, take their teacher labels from that directory instead.
    game_only: frozenset[str] = frozenset()
    game_labels: Path | None = None

    def __post_init__(self) -> None:
        if not self.shards:
            raise ValueError("a stream needs at least one shard")
        if self.batch_size <= 0 or self.readers <= 0:
            raise ValueError("batch_size and readers must be positive")
        if not 0.0 < self.sample_rate <= 1.0:
            raise ValueError("sample_rate must be in (0, 1]")
        if abs(1.0 / self.sample_rate - round(1.0 / self.sample_rate)) > 1e-9:
            raise ValueError("sample_rate must be 1/n for a whole number n (0.5, 0.25, ...)")
        if self.shuffle_buffer < self.batch_size:
            raise ValueError("shuffle_buffer must hold at least one batch")
        if not 0 <= self.worker < self.workers:
            raise ValueError("worker index out of range")
        if self.opening_plies < 0 or not 0.0 < self.opening_keep <= 1.0:
            raise ValueError("opening_plies must be >= 0 and opening_keep in (0, 1]")

    @property
    def period(self) -> int:
        return round(1.0 / self.sample_rate)


class _Stride:
    """Game indices owned by one worker: ``index % workers == worker``."""

    def __init__(self, worker: int, workers: int) -> None:
        self.worker, self.workers = worker, workers

    def __contains__(self, index: object) -> bool:
        return isinstance(index, int) and index % self.workers == self.worker


def _shard_games(path: Path, owned: _Stride) -> Iterator[tuple[str, str, bytes]]:
    for name, raw in iter_games(path, only=owned):
        yield path.name, name, raw


def _games(spec: StreamSpec, rng: random.Random) -> Iterator[tuple[str, int, Batch]]:
    """Yield (game name, pass number, positions) from ``readers`` open shards at once.

    With teacher labels, each game's targets are mixed with the teacher's before
    sampling, so sampled plies keep their own labels.
    """
    owned = _Stride(spec.worker, spec.workers)
    labels: dict[str, ShardLabels] = {}
    epoch = 0
    while True:
        order = list(spec.shards)
        rng.shuffle(order)
        pending = iter(order)
        active: list[Iterator[tuple[str, str, bytes]]] = []
        for _ in range(min(spec.readers, len(order))):
            active.append(_shard_games(next(pending), owned))
        while active:
            slot = rng.randrange(len(active))
            try:
                shard, name, raw = next(active[slot])
            except StopIteration:
                replacement = next(pending, None)
                if replacement is None:
                    active.pop(slot)
                else:
                    active[slot] = _shard_games(replacement, owned)
                continue
            label_dir = spec.game_labels if shard in spec.game_only else spec.teacher_labels
            if label_dir is not None and shard not in labels:
                labels[shard] = ShardLabels.open(label_dir, shard)
            if label_dir is not None and name in labels[shard].excluded:
                continue  # left unlabelled on purpose by the labeller (teacher.py)
            game = records_to_batch(parse_records(raw))
            if label_dir is not None:
                game = labels[shard].apply(
                    game, name, spec.teacher_weight, spec.teacher_temperature
                )
            yield name, epoch, game
        epoch += 1
        if not spec.cycle:
            return


def kept_plies(name: str, plies: int, epoch: int, period: int) -> np.ndarray:  # type: ignore[type-arg]
    """Plies whose per-ply hash class equals this pass's class (``epoch % period``).

    Each ply of a game gets a pseudo-random class in ``0..period-1`` from a hash of
    the game name and ply number, so a pass keeps a random-looking 1/period of the
    plies, both sides to move mixed, and ``period`` consecutive passes keep every ply
    exactly once. Keeping every period-th ply instead gave each game a single side to
    move per pass, and ab_m1_fixed's value loss trailed ab_m1's by about 0.045.
    """
    base = np.uint64(zlib.crc32(name.encode()))
    with np.errstate(over="ignore"):  # uint64 wrap-around is the intended mixing
        x = (np.arange(plies, dtype=np.uint64) + base) * np.uint64(0x9E3779B97F4A7C15)
        x ^= x >> np.uint64(31)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(29)
    return np.flatnonzero(x % np.uint64(period) == np.uint64(epoch % period))


class ShuffleBuffer:
    """Fixed-capacity random-replacement buffer with preallocated field arrays.

    Each incoming position either fills a free slot or evicts a uniformly random
    occupant, which is emitted. Cost per position is O(1), independent of capacity.
    """

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.size = 0
        self._rng = rng
        self._fields: Batch | None = None

    def _allocate(self, like: Batch) -> Batch:
        def empty(array: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
            return np.empty((self.capacity, *array.shape[1:]), dtype=array.dtype)

        return Batch(
            packed=empty(like.packed),
            aux=empty(like.aux),
            policy=empty(like.policy),
            wdl=empty(like.wdl),
            q_wdl=empty(like.q_wdl),
            plies_left=empty(like.plies_left),
        )

    @staticmethod
    def _write(dst: Batch, slots: np.ndarray, src: Batch) -> None:  # type: ignore[type-arg]
        dst.packed[slots] = src.packed
        dst.aux[slots] = src.aux
        dst.policy[slots] = src.policy
        dst.wdl[slots] = src.wdl
        dst.q_wdl[slots] = src.q_wdl
        dst.plies_left[slots] = src.plies_left

    def push(self, game: Batch) -> Batch | None:
        """Insert positions; return the evicted positions (or None while filling)."""
        if self._fields is None:
            self._fields = self._allocate(game)
        fields = self._fields
        free = min(self.capacity - self.size, len(game))
        if free:
            slots = np.arange(self.size, self.size + free)
            self._write(fields, slots, take(game, np.arange(free)))
            self.size += free
        if free == len(game):
            return None
        rest = take(game, np.arange(free, len(game)))
        victims = self._rng.choice(self.capacity, size=len(rest), replace=False)
        evicted = take(fields, victims)
        self._write(fields, victims, rest)
        return evicted

    def drain(self) -> Batch | None:
        """Remove and return every buffered position in random order."""
        if self._fields is None or self.size == 0:
            return None
        order = self._rng.permutation(self.size)
        out = take(self._fields, order)
        self.size = 0
        return out


def build_validation_set(
    shard: Path, positions: int, per_game: int, seed: int, teacher: ShardLabels | None = None
) -> Batch:
    """A fixed sample spread across the whole shard and across each game's phases.

    ``positions // per_game`` games are drawn uniformly (seeded) from every game in
    the shard. From each, ``per_game`` positions are taken at evenly spaced plies,
    so openings, middlegames and endgames are all represented. Deterministic for a
    given shard, size and seed. With ``teacher``, the same positions carry the
    teacher's targets instead of the game's; games the teacher's labels exclude are
    skipped, so such a sample can hold fewer games than it draws.
    """
    if positions <= 0 or per_game <= 0:
        raise ValueError("positions and per_game must be positive")
    total_games = count_games(shard)
    # 10% spare games: some games are shorter than per_game positions.
    wanted_games = min(total_games, -(-positions * 11 // (per_game * 10)))
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(total_games, size=wanted_games, replace=False).tolist())
    picked: list[Batch] = []
    for name, raw in iter_games(shard, only=chosen):
        if teacher is not None and name in teacher.excluded:
            continue
        game = records_to_batch(parse_records(raw))
        if teacher is not None:
            game = teacher.apply(game, name)
        count = min(per_game, len(game))
        plies = np.unique(np.linspace(0, len(game) - 1, count).round().astype(np.intp))
        picked.append(take(game, plies))
    merged = concat_batches(picked)
    if len(merged) < positions:
        raise ValueError(f"{shard.name} yields only {len(merged)} validation positions")
    order = rng.permutation(len(merged))[:positions]
    return take(merged, np.sort(order))


def stream_batches(spec: StreamSpec) -> Iterator[Batch]:
    """Yield shuffled batches of exactly ``batch_size`` positions.

    With ``cycle=False`` the final partial batch is dropped, so validation sees a
    deterministic set of full batches for a given spec.
    """
    rng = random.Random(spec.seed)
    np_rng = np.random.default_rng(spec.seed)
    buffer = ShuffleBuffer(spec.shuffle_buffer, np_rng)
    pending: list[Batch] = []
    pending_size = 0

    def emit(positions: Batch) -> Iterator[Batch]:
        nonlocal pending, pending_size
        pending.append(positions)
        pending_size += len(positions)
        if pending_size < spec.batch_size:
            return
        merged = concat_batches(pending)
        full = (len(merged) // spec.batch_size) * spec.batch_size
        for start in range(0, full, spec.batch_size):
            yield take(merged, np.arange(start, start + spec.batch_size))
        pending = [take(merged, np.arange(full, len(merged)))] if full < len(merged) else []
        pending_size = len(merged) - full

    for name, epoch, game in _games(spec, rng):
        keep = (
            kept_plies(name, len(game), epoch, spec.period)
            if spec.period > 1
            else np.arange(len(game))
        )
        if spec.opening_plies > 0 and spec.opening_keep < 1.0:
            late = keep >= spec.opening_plies
            keep = keep[late | (np_rng.random(len(keep)) < spec.opening_keep)]
        if keep.size == 0:
            continue
        if keep.size < len(game):
            game = take(game, keep)
        if spec.mirror:
            game = random_mirror(game, np_rng)
        evicted = buffer.push(game)
        if evicted is not None:
            yield from emit(evicted)
    remainder = buffer.drain()
    if remainder is not None:
        yield from emit(remainder)
