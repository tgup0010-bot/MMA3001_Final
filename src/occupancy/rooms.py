"""Friendly labels for the anonymous room identifiers used in the data.

Cross-referencing ``Sensor ID and Locations.xlsx`` against the occupancy
log's ``floorspaceid`` values resolved only 2 of the 5 rooms to a named
physical space (see the report, Limitations). The other 3 are left as
their raw identifier -- inventing a plausible-sounding name for them would
misrepresent what is actually known about the data.
"""

from __future__ import annotations

#: floorspaceid -> physical room name, for the 2 rooms that could be
#: traced through the location spreadsheet.
ROOM_LABELS: dict[str, str] = {
    "cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b": "G.20",
    "227f1068-9e41-469f-bef0-275f27a0bdba": "G.25 (Keenan Lab)",
}


def room_label(floorspaceid: str) -> str:
    """Return a human-friendly label for a room identifier.

    Args:
        floorspaceid: The raw room identifier from the occupancy log.

    Returns:
        The known physical room name if ``floorspaceid`` is in
        :data:`ROOM_LABELS`; otherwise ``"Zone <first 8 chars> (unidentified)"``
        -- never a made-up name.
    """
    if floorspaceid in ROOM_LABELS:
        return ROOM_LABELS[floorspaceid]
    return f"Zone {floorspaceid[:8]} (unidentified)"
