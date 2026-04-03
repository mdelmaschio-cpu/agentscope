# -*- coding: utf-8 -*-
"""Pydantic models for structured floor plan extraction output."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class _Room(BaseModel):
    """A single room in the floor plan."""

    name: str = Field(description="Room name (e.g. 'Living Room', 'Bedroom 1')")
    type: Literal[
        "living_room",
        "bedroom",
        "bathroom",
        "kitchen",
        "dining_room",
        "hallway",
        "storage",
        "balcony",
        "garage",
        "office",
        "other",
    ] = Field(description="Room functional type")
    area_sqm: float | None = Field(
        default=None,
        description="Estimated area in square metres. Null if not determinable.",
    )
    width_m: float | None = Field(
        default=None,
        description="Estimated width in metres. Null if not determinable.",
    )
    length_m: float | None = Field(
        default=None,
        description="Estimated length in metres. Null if not determinable.",
    )
    connected_to: list[str] = Field(
        default_factory=list,
        description="Names of rooms directly accessible from this room.",
    )
    notes: str | None = Field(
        default=None,
        description="Any relevant observations about the room (e.g. orientation, features).",
    )


class _Opening(BaseModel):
    """A door or archway connecting two spaces."""

    from_room: str = Field(description="Name of the first room.")
    to_room: str = Field(description="Name of the second room or 'exterior'.")
    type: Literal["door", "archway", "sliding_door", "french_door"] = Field(
        description="Type of opening.",
    )


class _WindowGroup(BaseModel):
    """Window count for a given room."""

    room: str = Field(description="Room name.")
    count: int = Field(description="Number of windows in this room.", ge=0)


class FloorPlan(BaseModel):
    """Structured representation of an extracted floor plan."""

    num_floors: int = Field(
        default=1,
        description="Number of floors visible in the provided images.",
        ge=1,
    )
    total_area_sqm: float | None = Field(
        default=None,
        description="Total estimated area across all floors in square metres.",
    )
    rooms: list[_Room] = Field(
        default_factory=list,
        description="All identified rooms.",
    )
    openings: list[_Opening] = Field(
        default_factory=list,
        description="Doors and archways connecting rooms.",
    )
    windows: list[_WindowGroup] = Field(
        default_factory=list,
        description="Window counts per room.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "Overall confidence in the extraction. "
            "'high' = clear architectural drawing, "
            "'medium' = photo of a plan or partial drawing, "
            "'low' = inferred from room photos with no explicit plan."
        ),
    )
    description: str = Field(
        description="A brief natural-language summary of the floor plan layout.",
    )
