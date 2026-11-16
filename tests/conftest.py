# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic V6 records and shards for tests (no Lc0 data is committed)."""

from __future__ import annotations

import gzip
import hashlib
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mouse_moe.policy_map import POLICY_SIZE, policy_index_of
from mouse_moe.v6 import PLANE_BYTES, RECORD_DTYPE

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "test80"

# Start position, white to move, from the side to move's perspective (plane: squares).
STARTPOS = {
    0: list(range(8, 16)),
    1: [1, 6],
    2: [2, 5],
    3: [0, 7],
    4: [3],
    5: [4],
    6: list(range(48, 56)),
    7: [57, 62],
    8: [58, 61],
    9: [56, 63],
    10: [59],
    11: [60],
}
STARTPOS_MOVES = [f"{f}2{f}{r}" for f in "abcdefgh" for r in "34"] + [
    "b1a3",
    "b1c3",
    "g1f3",
    "g1h3",
]


def pack_planes(planes: dict[int, list[int]]) -> npt.NDArray[np.uint8]:
    packed = np.zeros(PLANE_BYTES, dtype=np.uint8)
    for plane, squares in planes.items():
        for square in squares:
            packed[plane * 8 + square // 8] |= 0x80 >> (square % 8)
    return packed


def make_record(**fields: object) -> npt.NDArray[np.void]:
    record = np.zeros(1, dtype=RECORD_DTYPE)
    record["version"] = 6
    record["input_format"] = 1
    probabilities = np.full(POLICY_SIZE, -1.0, dtype=np.float32)
    for move in STARTPOS_MOVES:
        probabilities[policy_index_of()[move]] = 1.0 / len(STARTPOS_MOVES)
    record["probabilities"] = probabilities
    record["planes"] = pack_planes(STARTPOS)
    for name in ("castling_us_ooo", "castling_us_oo", "castling_them_ooo", "castling_them_oo"):
        record[name] = 1
    record["result_q"] = 0.2
    record["result_d"] = 0.5
    record["best_q"] = 0.1
    record["best_d"] = 0.6
    record["plies_left"] = 80.0
    record["played_idx"] = policy_index_of()["e2e4"]
    for name, value in fields.items():
        record[name] = value
    return record


def write_shard(directory: Path, name: str, games: list[npt.NDArray[np.void]]) -> Path:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        directory_entry = tarfile.TarInfo("games")
        directory_entry.type = tarfile.DIRTYPE
        archive.addfile(directory_entry)
        for index, game in enumerate(games):
            payload = gzip.compress(game.tobytes())
            info = tarfile.TarInfo(f"games/game_{index:05d}.gz")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    path = directory / name
    path.write_bytes(buffer.getvalue())
    return path


def write_manifest(directory: Path) -> None:
    rows = ['"file","bytes","sha256"']
    for path in sorted(directory.glob("*.tar")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f'"{path.name}","{path.stat().st_size}","{digest}"')
    (directory / "MANIFEST.csv").write_text("\n".join(rows) + "\n", encoding="ascii")


def numbered_game(start: int, length: int) -> npt.NDArray[np.void]:
    """A game whose plies_left values are unique labels start..start+length-1."""
    return np.concatenate([make_record(plies_left=float(start + i)) for i in range(length)])


@pytest.fixture
def synthetic_shards(tmp_path: Path) -> Path:
    """Three shards x 5 games x 10 positions, every position uniquely labelled."""
    label = 0
    for shard in range(3):
        games = []
        for _ in range(5):
            games.append(numbered_game(label, 10))
            label += 10
        write_shard(tmp_path, f"shard-{shard}.tar", games)
    write_manifest(tmp_path)
    return tmp_path
