from enum import Enum
from typing import TypedDict


class ClaimType(Enum):
    STRING = 1
    TIME = 2
    ENTITY = 3
    QUANTITY = 4
    LEXEME = 5
    PROPERTY = 6
    # GLOBE_COORDINATES = 5
    # MONOLINGUAL_TEXT = 6
    # MULTILINGUAL_TEXT = 7
    # URL = 8
    # EXTERNAL_ID = 9
    UNKNOWN = 10


Entity = TypedDict("Entity", {"id": str, "claims": dict["Entity"]})

Claim = TypedDict("Claim", {"type": ClaimType, "value": str})

DiscoveredEntity = TypedDict("DiscoveredEntity", {"candidate": str, "index": int})
