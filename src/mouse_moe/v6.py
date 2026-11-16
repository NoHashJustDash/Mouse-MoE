# SPDX-License-Identifier: GPL-3.0-or-later
"""Lc0 ``V6TrainingData`` records -> Mouse engine input planes and targets.

Record layout: lc0 v0.32.1 ``src/trainingdata/trainingdata_v6.h`` (8356 bytes,
packed, little-endian). Input planes reproduce the Mouse engine's classical
input-format-1 encoder (``engine/nn/encoder.cpp``) bit for bit:
``planes[p * 64 + square]`` with a1 = square 0; planes 0..103 are eight history
steps of 13 planes (own P N B R Q K, opponent P N B R Q K, repetition); 104..107
castling (our O-O-O, our O-O, their O-O-O, their O-O); 108 black to move; 109
the *raw* rule-50 count; 110 zero; 111 ones.

lc0 stores each training bitboard with the bits of every byte reversed, so an
MSB-first unpack of the 832 plane bytes yields squares in a1..h8 order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mouse_moe.policy_map import POLICY_SIZE

RECORD_SIZE = 8356
SUPPORTED_VERSION = 6
SUPPORTED_INPUT_FORMAT = 1
PIECE_PLANES = 104
PLANE_BYTES = PIECE_PLANES * 8
INPUT_PLANES = 112
AUX_FEATURES = 8  # values broadcast into planes 104..111

RECORD_DTYPE = np.dtype(
    [
        ("version", "<u4"),
        ("input_format", "<u4"),
        ("probabilities", "<f4", (POLICY_SIZE,)),
        ("planes", "u1", (PLANE_BYTES,)),
        ("castling_us_ooo", "u1"),
        ("castling_us_oo", "u1"),
        ("castling_them_ooo", "u1"),
        ("castling_them_oo", "u1"),
        ("side_to_move_or_enpassant", "u1"),
        ("rule50_count", "u1"),
        ("invariance_info", "u1"),
        ("dummy", "u1"),
        ("root_q", "<f4"),
        ("best_q", "<f4"),
        ("root_d", "<f4"),
        ("best_d", "<f4"),
        ("root_m", "<f4"),
        ("best_m", "<f4"),
        ("plies_left", "<f4"),
        ("result_q", "<f4"),
        ("result_d", "<f4"),
        ("played_q", "<f4"),
        ("played_d", "<f4"),
        ("played_m", "<f4"),
        ("orig_q", "<f4"),
        ("orig_d", "<f4"),
        ("orig_m", "<f4"),
        ("visits", "<u4"),
        ("played_idx", "<u2"),
        ("best_idx", "<u2"),
        ("policy_kld", "<f4"),
        ("q_st", "<f4"),
    ]
)
if RECORD_DTYPE.itemsize != RECORD_SIZE:
    raise AssertionError(f"V6 dtype is {RECORD_DTYPE.itemsize} bytes, expected {RECORD_SIZE}")


class RecordError(ValueError):
    """A training record is malformed or uses an unsupported format."""


def parse_records(raw: bytes) -> npt.NDArray[np.void]:
    """Parse one decompressed game into records, validating version and format."""
    if len(raw) == 0 or len(raw) % RECORD_SIZE != 0:
        raise RecordError(f"game length {len(raw)} is not a positive multiple of {RECORD_SIZE}")
    records = np.frombuffer(raw, dtype=RECORD_DTYPE)
    if np.any(records["version"] != SUPPORTED_VERSION):
        raise RecordError(f"unsupported record version(s): {np.unique(records['version'])}")
    if np.any(records["input_format"] != SUPPORTED_INPUT_FORMAT):
        raise RecordError(f"unsupported input format(s): {np.unique(records['input_format'])}")
    return records


@dataclass(frozen=True, slots=True)
class Batch:
    """Compact training batch; planes are expanded on the GPU (see ``model.expand_planes``).

    packed:      uint8 [N, 832]  MSB-first bitboards for planes 0..103
    aux:         float32 [N, 8]  values for planes 104..111
    policy:      float32 [N, 1858] visit distribution, negative for illegal moves
    wdl:         float32 [N, 3]  game result as win/draw/loss
    q_wdl:       float32 [N, 3]  search best-move Q/D as win/draw/loss
    plies_left:  float32 [N]
    """

    packed: npt.NDArray[np.uint8]
    aux: npt.NDArray[np.float32]
    policy: npt.NDArray[np.float32]
    wdl: npt.NDArray[np.float32]
    q_wdl: npt.NDArray[np.float32]
    plies_left: npt.NDArray[np.float32]

    def __len__(self) -> int:
        return int(self.packed.shape[0])


def _to_wdl(q: npt.NDArray[np.float32], d: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    win = 0.5 * (1.0 - d + q)
    loss = 0.5 * (1.0 - d - q)
    wdl = np.stack([win, d, loss], axis=1).astype(np.float32)
    # Float noise from (q, d) can push a component a hair below zero.
    np.clip(wdl, 0.0, 1.0, out=wdl)
    return wdl


def records_to_batch(records: npt.NDArray[np.void]) -> Batch:
    aux = np.zeros((len(records), AUX_FEATURES), dtype=np.float32)
    aux[:, 0] = records["castling_us_ooo"] != 0
    aux[:, 1] = records["castling_us_oo"] != 0
    aux[:, 2] = records["castling_them_ooo"] != 0
    aux[:, 3] = records["castling_them_oo"] != 0
    aux[:, 4] = records["side_to_move_or_enpassant"] != 0
    aux[:, 5] = records["rule50_count"].astype(np.float32)
    aux[:, 6] = 0.0
    aux[:, 7] = 1.0
    return Batch(
        packed=np.ascontiguousarray(records["planes"], dtype=np.uint8),
        aux=aux,
        policy=np.ascontiguousarray(records["probabilities"], dtype=np.float32),
        wdl=_to_wdl(records["result_q"], records["result_d"]),
        q_wdl=_to_wdl(records["best_q"], records["best_d"]),
        plies_left=np.ascontiguousarray(records["plies_left"], dtype=np.float32),
    )


def concat_batches(batches: list[Batch]) -> Batch:
    return Batch(
        packed=np.concatenate([b.packed for b in batches]),
        aux=np.concatenate([b.aux for b in batches]),
        policy=np.concatenate([b.policy for b in batches]),
        wdl=np.concatenate([b.wdl for b in batches]),
        q_wdl=np.concatenate([b.q_wdl for b in batches]),
        plies_left=np.concatenate([b.plies_left for b in batches]),
    )


def take(batch: Batch, index: npt.NDArray[np.intp]) -> Batch:
    return Batch(
        packed=batch.packed[index],
        aux=batch.aux[index],
        policy=batch.policy[index],
        wdl=batch.wdl[index],
        q_wdl=batch.q_wdl[index],
        plies_left=batch.plies_left[index],
    )


def expand_planes(
    packed: npt.NDArray[np.uint8], aux: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Reference (NumPy) expansion to the engine's dense [N, 112, 8, 8] float32 input."""
    count = packed.shape[0]
    bits = np.unpackbits(packed, axis=1, bitorder="big").reshape(count, PIECE_PLANES, 64)
    planes = np.empty((count, INPUT_PLANES, 64), dtype=np.float32)
    planes[:, :PIECE_PLANES] = bits
    planes[:, PIECE_PLANES:] = aux[:, :, None]
    return planes.reshape(count, INPUT_PLANES, 8, 8)


def piece_bitboards(packed_row: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
    """[104, 64] boolean occupancy for one record (validation helper)."""
    bits = np.unpackbits(packed_row, bitorder="big").reshape(PIECE_PLANES, 64)
    return bits.astype(bool)
