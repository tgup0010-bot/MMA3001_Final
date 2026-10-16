"""Unit tests for occupancy.rooms.room_label."""

from occupancy.rooms import ROOM_LABELS, room_label


def test_known_room_returns_friendly_name():
    known_id = next(iter(ROOM_LABELS))
    assert room_label(known_id) == ROOM_LABELS[known_id]


def test_unknown_room_returns_placeholder_not_a_made_up_name():
    label = room_label("00000000-1111-2222-3333-444444444444")
    assert label == "Zone 00000000 (unidentified)"
