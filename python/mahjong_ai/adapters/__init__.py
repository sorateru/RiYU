"""External observation adapters.

Adapters stop at the public ``GameEvent`` contract.  They never mutate the
authoritative state or feed service-specific objects into the model.
"""

from .majsoul import (
    MAJSOUL_EVENT_PROFILE,
    MahjongSoulAdapterError,
    MahjongSoulEvent,
    map_majsoul_event,
    parse_majsoul_event,
)

__all__ = [
    "MAJSOUL_EVENT_PROFILE",
    "MahjongSoulAdapterError",
    "MahjongSoulEvent",
    "map_majsoul_event",
    "parse_majsoul_event",
]
