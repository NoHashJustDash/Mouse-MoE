# SPDX-License-Identifier: GPL-3.0-or-later
"""Teacher-network labels for distillation: storage, lookup and target replacement.

A teacher (Lc0 BT4) labels every position of a training shard once. The loader then
replaces each game's policy, value and moves-left targets with the teacher's, so the
losses and the training loop are unchanged.

Files per shard ``S`` in the label directory, rows in archive order of games and plies:

    S.idx.npy    uint16  [N, K]  teacher's K most likely legal moves (Lc0 policy indices)
    S.prob.npy   float16 [N, K]  their probabilities (softmax over the legal moves)
    S.wdl.npy    float16 [N, 3]  teacher win/draw/loss probabilities
    S.mlh.npy    float16 [N]     teacher moves-left estimate (plies)
    S.games.json member names and row offsets, and the games left unlabelled on purpose
    S.done.json  written last: teacher SHA-256, row count, how the labels were made

Only shards with a ``done`` file are complete; anything else is refused. A game listed
as excluded (for example, one the search labeller cannot replay) is skipped by the
loader; any other game without labels is an error.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mouse_moe.v6 import Batch

TOP_K = 16  # teacher moves kept per position; the labeler reports the mass they cover
_ARRAYS = ("idx", "prob", "wdl", "mlh")


class TeacherLabelError(ValueError):
    """Teacher labels are missing, incomplete or do not match the shard."""


def _atomic_save(path: Path, array: npt.NDArray[np.generic]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array)
    os.replace(temporary, path)


def write_shard_labels(
    directory: Path,
    shard: str,
    names: list[str],
    lengths: list[int],
    arrays: dict[str, npt.NDArray[np.generic]],
    teacher_sha256: str,
    excluded: list[str] | None = None,
    method: dict[str, object] | None = None,
) -> None:
    """Write one shard's labels; the ``done`` file goes last, so a crash leaves no done.

    ``excluded`` names games of the shard that deliberately have no labels; ``method``
    records how the labels were made (for example, the search settings).
    """
    excluded = list(excluded or [])
    overlap = set(excluded).intersection(names)
    if overlap:
        raise TeacherLabelError(f"{shard}: games both labelled and excluded: {sorted(overlap)[:3]}")
    rows = int(sum(lengths))
    expected = {"idx": (rows, TOP_K), "prob": (rows, TOP_K), "wdl": (rows, 3), "mlh": (rows,)}
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise TeacherLabelError(f"{shard}: {key} has shape {arrays[key].shape}, want {shape}")
    directory.mkdir(parents=True, exist_ok=True)
    done = directory / f"{shard}.done.json"
    done.unlink(missing_ok=True)
    dtypes = {"idx": np.uint16, "prob": np.float16, "wdl": np.float16, "mlh": np.float16}
    for key in _ARRAYS:
        _atomic_save(directory / f"{shard}.{key}.npy", arrays[key].astype(dtypes[key]))
    offsets = np.concatenate([[0], np.cumsum(lengths)]).tolist()
    games = directory / f"{shard}.games.json"
    games.with_name(games.name + ".tmp").write_text(
        json.dumps({"names": names, "offsets": offsets, "excluded": excluded}), encoding="utf-8"
    )
    os.replace(games.with_name(games.name + ".tmp"), games)
    meta: dict[str, object] = {
        "teacher_sha256": teacher_sha256,
        "rows": rows,
        "games": len(names),
        "excluded_games": len(excluded),
    }
    if method is not None:
        meta["method"] = method
    done.write_text(json.dumps(meta), encoding="utf-8")


class LabelRows:
    """One ``.npy`` label array, read by row range with ordinary file reads.

    Not memory-mapped: random sampling touches nearly every page of a mapped file, and
    the touched pages stay in the data-loader workers' working sets, where Windows does
    not count them as available RAM. Over the 22-shard M1 run (13 GB of labels) that
    sank available RAM to the training memory guard's 4 GiB reserve in about 1.5 hours
    (2026-10-26). Read pages go to the file cache, which Windows counts as available.
    """

    def __init__(self, path: Path) -> None:
        self._file = path.open("rb")
        # numpy.lib.format ships without type annotations.
        version = np.lib.format.read_magic(self._file)  # type: ignore[no-untyped-call]
        readers = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }
        if version not in readers:
            raise TeacherLabelError(f"{path}: unsupported .npy version {version}")
        shape, fortran_order, dtype = readers[version](self._file)  # type: ignore[no-untyped-call]
        if fortran_order or not shape:
            raise TeacherLabelError(f"{path}: expected a C-ordered array with rows")
        self.path = path
        self.shape: tuple[int, ...] = tuple(shape)
        self.dtype = np.dtype(dtype)
        self._data_offset = self._file.tell()
        self._row_items = math.prod(self.shape[1:])

    def __len__(self) -> int:
        return self.shape[0]

    def read(self, start: int, length: int) -> npt.NDArray[np.generic]:
        """Rows ``start : start + length`` as a new array."""
        if start < 0 or start + length > len(self):
            raise TeacherLabelError(f"{self.path}: rows {start}:{start + length} of {len(self)}")
        self._file.seek(self._data_offset + start * self._row_items * self.dtype.itemsize)
        data = np.fromfile(self._file, dtype=self.dtype, count=length * self._row_items)
        if data.size != length * self._row_items:
            raise TeacherLabelError(f"{self.path}: file ends before row {start + length}")
        return data.reshape((length, *self.shape[1:]))


@dataclass
class ShardLabels:
    """Labels of one shard, read from disk per game (see ``LabelRows``)."""

    shard: str
    teacher_sha256: str
    offsets: dict[str, tuple[int, int]]
    idx: LabelRows
    prob: LabelRows
    wdl: LabelRows
    mlh: LabelRows
    excluded: frozenset[str] = frozenset()

    @classmethod
    def open(cls, directory: Path, shard: str) -> ShardLabels:
        done = directory / f"{shard}.done.json"
        if not done.is_file():
            raise TeacherLabelError(f"no complete teacher labels for {shard} in {directory}")
        meta = json.loads(done.read_text(encoding="utf-8"))
        games = json.loads((directory / f"{shard}.games.json").read_text(encoding="utf-8"))
        bounds = games["offsets"]
        offsets = {
            name: (bounds[i], bounds[i + 1] - bounds[i]) for i, name in enumerate(games["names"])
        }
        arrays = {key: LabelRows(directory / f"{shard}.{key}.npy") for key in _ARRAYS}
        if any(len(array) != meta["rows"] for array in arrays.values()):
            raise TeacherLabelError(f"{shard}: label arrays do not match the done file")
        excluded = frozenset(games.get("excluded", []))
        if len(excluded) != meta.get("excluded_games", 0):
            raise TeacherLabelError(f"{shard}: excluded games do not match the done file")
        return cls(shard, meta["teacher_sha256"], offsets, **arrays, excluded=excluded)

    def apply(self, game: Batch, name: str, weight: float = 1.0, temperature: float = 1.0) -> Batch:
        """``game`` with its targets mixed with the teacher's.

        Each target becomes ``weight * teacher + (1 - weight) * game``: policy against
        the search policy, WDL against the game result, Q-WDL against the search value,
        moves-left against the plies actually left. Cross-entropy is linear in the
        target, so this equals mixing the two losses. The teacher policy is sharpened
        or softened as ``p ** (1 / temperature)``. weight 1, temperature 1 replaces the
        targets with the teacher's exactly.
        """
        if not 0.0 <= weight <= 1.0 or temperature <= 0.0:
            raise TeacherLabelError("teacher weight must be in [0, 1] and temperature positive")
        if name not in self.offsets:
            raise TeacherLabelError(f"{self.shard}: no labels for game {name}")
        start, length = self.offsets[name]
        if length != len(game):
            raise TeacherLabelError(
                f"{self.shard}/{name}: {length} labelled plies, game has {len(game)}"
            )
        idx = self.idx.read(start, length).astype(np.intp)
        prob = self.prob.read(start, length).astype(np.float32)
        if temperature != 1.0:
            prob = np.where(prob > 0, prob ** np.float32(1.0 / temperature), 0.0).astype(np.float32)
        legal = game.policy >= 0
        policy = np.where(legal, 0.0, -1.0).astype(np.float32)
        ply, column = np.nonzero(prob > 0)
        targets = idx[ply, column]
        if not legal[ply, targets].all():
            raise TeacherLabelError(f"{self.shard}/{name}: teacher label on an illegal move")
        policy[ply, targets] = prob[ply, column]
        mass = np.where(legal, policy, 0.0).sum(axis=1, keepdims=True)
        if (mass <= 0).any():
            raise TeacherLabelError(f"{self.shard}/{name}: a position has no teacher policy")
        policy = np.where(legal, policy / mass, 0.0)
        wdl = self.wdl.read(start, length).astype(np.float32)
        plies_left = self.mlh.read(start, length).astype(np.float32)
        own = np.float32(1.0 - weight)
        mix = np.float32(weight)
        game_policy = np.where(legal, game.policy, 0.0)
        return Batch(
            packed=game.packed,
            aux=game.aux,
            policy=np.where(legal, mix * policy + own * game_policy, -1.0).astype(np.float32),
            wdl=(mix * wdl + own * game.wdl).astype(np.float32),
            q_wdl=(mix * wdl + own * game.q_wdl).astype(np.float32),
            plies_left=(mix * plies_left + own * game.plies_left).astype(np.float32),
        )


def require_labels(directory: Path, shards: list[Path]) -> None:
    """Fail before training starts if any shard lacks complete labels."""
    missing = [s.name for s in shards if not (directory / f"{s.name}.done.json").is_file()]
    if missing:
        raise TeacherLabelError(f"teacher labels missing in {directory} for: {', '.join(missing)}")
