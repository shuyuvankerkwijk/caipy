"""
Antenna enum.
"""

from enum import Enum
from typing import Union

class Antenna(Enum):
    NORTH = 1
    SOUTH = 2

def antenna_letter(ant: "Antenna") -> str:
    """Return short letter code ('N' or 'S') for an antenna enum.

    Args:
        ant: Antenna enum member

    Returns:
        'N' for Antenna.NORTH, 'S' for Antenna.SOUTH
    """
    if ant == Antenna.NORTH:
        return "N"
    if ant == Antenna.SOUTH:
        return "S"
    raise ValueError(f"Unknown antenna enum: {ant}")

def parse_antenna(value: Union[str, "Antenna"]) -> "Antenna":
    """Parse input into an Antenna enum.

    Accepts Antenna enum (returned as-is), letter codes ('N'/'S'), and
    common lowercase/uppercase names ('north'/'south').
    """
    if isinstance(value, Antenna):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("n", "north"):
            return Antenna.NORTH
        if v in ("s", "south"):
            return Antenna.SOUTH
    raise ValueError(f"Invalid antenna identifier: {value}")