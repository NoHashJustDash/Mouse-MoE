# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pytest

from conftest import STARTPOS, make_record
from mouse_moe.v6 import (
    RECORD_DTYPE,
    RECORD_SIZE,
    RecordError,
    expand_planes,
    parse_records,
    records_to_batch,
)


def test_dtype_matches_lc0_struct() -> None:
    assert RECORD_DTYPE.itemsize == RECORD_SIZE == 8356
    assert RECORD_DTYPE.fields is not None
    assert RECORD_DTYPE.fields["played_idx"][1] == 8344
    assert RECORD_DTYPE.fields["q_st"][1] == 8352


def test_planes_follow_engine_layout() -> None:
    record = make_record(rule50_count=37, side_to_move_or_enpassant=1, castling_them_oo=0)
    batch = records_to_batch(parse_records(record.tobytes()))
    planes = expand_planes(batch.packed, batch.aux).reshape(112, 64)
    for plane, squares in STARTPOS.items():
        assert sorted(np.flatnonzero(planes[plane]).tolist()) == squares
    assert planes[12:104].sum() == 0
    assert planes[104:107].min() == 1 and planes[107].max() == 0  # their O-O cleared
    assert planes[108].min() == 1  # black to move
    assert np.all(planes[109] == 37.0)  # raw rule-50, as the engine encodes it
    assert planes[110].max() == 0 and planes[111].min() == 1


def test_targets() -> None:
    batch = records_to_batch(parse_records(make_record().tobytes()))
    np.testing.assert_allclose(batch.wdl[0], [0.35, 0.5, 0.15], atol=1e-6)
    np.testing.assert_allclose(batch.q_wdl[0], [0.25, 0.6, 0.15], atol=1e-6)
    assert batch.plies_left[0] == 80.0
    assert abs(batch.policy[0][batch.policy[0] > 0].sum() - 1.0) < 1e-6


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"", "multiple"),
        (b"x" * 100, "multiple"),
        (make_record(version=5).tobytes(), "version"),
        (make_record(input_format=3).tobytes(), "input format"),
    ],
    ids=["empty", "partial-record", "version-5", "input-format-3"],
)
def test_rejects_malformed(raw: bytes, message: str) -> None:
    with pytest.raises(RecordError, match=message):
        parse_records(raw)
