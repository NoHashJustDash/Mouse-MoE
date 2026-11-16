# SPDX-License-Identifier: GPL-3.0-or-later
"""Structural validation of decoded training data against chess geometry.

Proves that plane decoding and the 1858 policy map agree with each other on real
records: every move with positive visit probability must start on a square that
holds one of our pieces, knight-geometry moves must be made by a knight,
promotions by a pawn on rank 7, and so on. A mirrored or mis-ordered plane decode
or policy table fails these checks immediately.

Usage: python -I -m mouse_moe.validate_data --data DIR --games-per-shard 200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mouse_moe.policy_map import POLICY_SIZE, decode
from mouse_moe.shards import iter_games, verified_shards
from mouse_moe.v6 import parse_records, piece_bitboards, records_to_batch

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
BACK_RANKS = set(range(8)) | set(range(56, 64))


def _knight_geometry(source: int, target: int) -> bool:
    return {abs(source % 8 - target % 8), abs(source // 8 - target // 8)} == {1, 2}


@dataclass
class Report:
    games: int = 0
    positions: int = 0
    failures: Counter[str] = field(default_factory=Counter)
    castling_encoding: Counter[str] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)
    policy_mass_min: float = 1.0
    policy_mass_max: float = 0.0

    def fail(self, kind: str, detail: str) -> None:
        self.failures[kind] += 1
        if len(self.examples) < 20:
            self.examples.append(f"{kind}: {detail}")

    def to_dict(self) -> dict[str, object]:
        return {
            "games": self.games,
            "positions": self.positions,
            "failures": dict(self.failures),
            "castling_encoding": dict(self.castling_encoding),
            "policy_mass_range": [self.policy_mass_min, self.policy_mass_max],
            "examples": self.examples,
        }


def _check_move(
    report: Report,
    ours: npt.NDArray[np.bool_],
    theirs: npt.NDArray[np.bool_],
    index: int,
    where: str,
) -> None:
    source, target, promotion = decode(index)
    own_piece = [p for p in range(6) if ours[p, source]]
    if not own_piece:
        report.fail("move_from_empty", f"{where} idx={index}")
        return
    piece = own_piece[0]
    if ours[:, target].any():
        if piece == KING and ours[ROOK, target] and source // 8 == 0 and target // 8 == 0:
            report.castling_encoding["king_to_rook"] += 1
        else:
            report.fail("move_onto_own_piece", f"{where} idx={index}")
        return
    if piece == KING and abs(source - target) == 2 and source // 8 == 0 == target // 8:
        report.castling_encoding["king_two_squares"] += 1
    if _knight_geometry(source, target) and piece != KNIGHT:
        report.fail("knight_move_by_non_knight", f"{where} idx={index} piece={piece}")
    if promotion and piece != PAWN:
        report.fail("promotion_by_non_pawn", f"{where} idx={index} piece={piece}")
    if piece == PAWN:
        forward = target - source
        if forward in (7, 9) and not theirs[:, target].any():
            report.castling_encoding["pawn_diagonal_to_empty(en_passant)"] += 1
        elif forward not in (7, 8, 9, 16):
            report.fail("illegal_pawn_geometry", f"{where} idx={index} delta={forward}")


def validate_records(raw: bytes, report: Report, where: str) -> None:
    records = parse_records(raw)
    batch = records_to_batch(records)
    report.games += 1
    report.positions += len(records)
    for row in range(len(records)):
        loc = f"{where}#{row}"
        boards = piece_bitboards(batch.packed[row])
        ours, theirs = boards[0:6], boards[6:12]
        if ours[KING].sum() != 1 or theirs[KING].sum() != 1:
            report.fail("king_count", loc)
        if (boards[0:12].sum(axis=0) > 1).any():
            report.fail("overlapping_pieces", loc)
        pawns = np.flatnonzero(ours[PAWN] | theirs[PAWN])
        if BACK_RANKS.intersection(int(p) for p in pawns):
            report.fail("pawn_on_back_rank", loc)
        policy = batch.policy[row]
        positive = np.flatnonzero(policy > 0)
        mass = float(policy[positive].sum())
        report.policy_mass_min = min(report.policy_mass_min, mass)
        report.policy_mass_max = max(report.policy_mass_max, mass)
        if abs(mass - 1.0) > 1e-3:
            report.fail("policy_mass", f"{loc} mass={mass}")
        for index in positive:
            _check_move(report, ours, theirs, int(index), loc)
        played = int(records["played_idx"][row])
        if not 0 <= played < POLICY_SIZE or policy[played] < 0:
            report.fail("played_move_illegal", f"{loc} idx={played}")
        else:
            _check_move(report, ours, theirs, played, loc + "(played)")
        wdl = batch.wdl[row]
        if abs(float(wdl.sum()) - 1.0) > 1e-4:
            report.fail("wdl_sum", f"{loc} {wdl}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--games-per-shard", type=int, default=200)
    parser.add_argument("--shards", nargs="*", help="shard file names (default: all in manifest)")
    parser.add_argument("--json", type=Path, help="write the report as JSON")
    args = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("run with python -I: training data is untrusted input")

    from mouse_moe.shards import load_manifest

    names = args.shards or sorted(load_manifest(args.data))
    report = Report()
    for path in verified_shards(args.data, names):
        for count, (member, raw) in enumerate(iter_games(path)):
            if count >= args.games_per_shard:
                break
            validate_records(raw, report, f"{path.name}:{member}")
    result = report.to_dict()
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if not report.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
