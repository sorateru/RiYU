import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("mahjong")

from mahjong.shanten import Shanten
from tools.validate_riyu_lake import has_source_turn_order_conflict, structural_standard


def test_structural_quad_can_have_no_physical_completion():
    hand = [0] * 34
    for kind in (3, 3, 3, 3, 5, 6, 7, 12, 13, 14):
        hand[kind] += 1
    reference = Shanten()
    assert reference.calculate_shanten_for_regular_hand(hand) == 1
    assert structural_standard(reference, hand, fixed=1) == 0
    for kind, n in enumerate(hand):
        if n < 4:
            drawn = hand.copy()
            drawn[kind] += 1
            assert structural_standard(reference, drawn, fixed=1) >= 0


def test_own_discard_before_self_kan_is_a_source_order_conflict():
    draw = {"type": "draw", "seat": 1}
    discard = {"type": "discard", "seat": 1}
    kan = {"type": "call", "seat": 1, "meld": {"kind": "ankan"}}
    assert has_source_turn_order_conflict([draw, discard, kan])
    assert not has_source_turn_order_conflict([draw, kan, draw, discard])
    pon = {"type": "call", "seat": 2, "meld": {"kind": "pon"}}
    assert not has_source_turn_order_conflict([draw, discard, pon])
