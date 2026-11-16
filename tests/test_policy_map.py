# SPDX-License-Identifier: GPL-3.0-or-later
from mouse_moe.policy_map import (
    KNIGHT_CHANNEL,
    ORDINARY_SIZE,
    POLICY_SIZE,
    attention_gather_tables,
    decode,
    policy_index_of,
    policy_moves,
    square_index,
    square_name,
)


def test_table_size_and_uniqueness() -> None:
    moves = policy_moves()
    assert len(moves) == POLICY_SIZE == 1858
    assert len(set(moves)) == POLICY_SIZE
    assert sum(len(m) == 4 for m in moves) == ORDINARY_SIZE == 1792


def test_known_lc0_ordering() -> None:
    moves = policy_moves()
    # Source-major, then target square order, queen lines + knight jumps.
    assert moves[:10] == (
        "a1b1",
        "a1c1",
        "a1d1",
        "a1e1",
        "a1f1",
        "a1g1",
        "a1h1",
        "a1a2",
        "a1b2",
        "a1c2",
    )
    assert moves[ORDINARY_SIZE : ORDINARY_SIZE + 3] == ("a7a8q", "a7a8r", "a7a8b")
    assert moves[-3:] == ("h7h8q", "h7h8r", "h7h8b")
    # Castling is encoded king -> own rook (verified on 786,770 real positions).
    assert "e1h1" in policy_index_of() and "e1a1" in policy_index_of()


def test_squares_round_trip() -> None:
    for square in range(64):
        assert square_index(square_name(square)) == square


def test_gather_tables() -> None:
    base, promo = attention_gather_tables()
    index = policy_index_of()
    a7a8 = index["a7a8"]
    assert base[a7a8] == 48 * 64 + 56 and promo[a7a8] == 0 * 4 + KNIGHT_CHANNEL
    assert (
        promo[index["a7a8q"]] == 0
        and promo[index["b7a8r"]] == 1
        and promo[index["h7h8b"]] == 7 * 4 + 2
    )
    assert promo[index["e2e4"]] == 32 and base[index["e2e4"]] == 12 * 64 + 28
    for i in range(POLICY_SIZE):
        source, target, _ = decode(i)
        assert base[i] == source * 64 + target
