# -*- coding: utf-8 -*-
"""Unit tests for the floor plan extractor example."""
import base64
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import ToolUseBlock
from agentscope.model import ChatModelBase, ChatResponse

# Make the example package importable from the tests directory
sys.path.insert(
    0,
    str(
        Path(__file__).parent.parent
        / "examples"
        / "agent"
        / "floor_plan_extractor"
    ),
)

from _extractor import FloorPlanExtractor  # noqa: E402
from _models import FloorPlan  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal fake floor plan payload that satisfies the FloorPlan schema
# ---------------------------------------------------------------------------
_FAKE_FLOOR_PLAN: dict[str, Any] = {
    "num_floors": 1,
    "total_area_sqm": 80.0,
    "rooms": [
        {
            "name": "Living Room",
            "type": "living_room",
            "area_sqm": 25.0,
            "width_m": 5.0,
            "length_m": 5.0,
            "connected_to": ["Kitchen", "Hallway"],
            "notes": None,
        },
        {
            "name": "Kitchen",
            "type": "kitchen",
            "area_sqm": 15.0,
            "width_m": 3.0,
            "length_m": 5.0,
            "connected_to": ["Living Room"],
            "notes": None,
        },
        {
            "name": "Hallway",
            "type": "hallway",
            "area_sqm": 8.0,
            "width_m": 1.5,
            "length_m": None,
            "connected_to": ["Living Room", "Bedroom 1", "Bathroom"],
            "notes": None,
        },
        {
            "name": "Bedroom 1",
            "type": "bedroom",
            "area_sqm": 20.0,
            "width_m": 4.0,
            "length_m": 5.0,
            "connected_to": ["Hallway"],
            "notes": None,
        },
        {
            "name": "Bathroom",
            "type": "bathroom",
            "area_sqm": 6.0,
            "width_m": 2.0,
            "length_m": 3.0,
            "connected_to": ["Hallway"],
            "notes": None,
        },
    ],
    "openings": [
        {
            "from_room": "Living Room",
            "to_room": "Kitchen",
            "type": "archway",
        },
        {
            "from_room": "Living Room",
            "to_room": "Hallway",
            "type": "door",
        },
        {
            "from_room": "Hallway",
            "to_room": "Bedroom 1",
            "type": "door",
        },
        {
            "from_room": "Hallway",
            "to_room": "Bathroom",
            "type": "door",
        },
        {
            "from_room": "Living Room",
            "to_room": "exterior",
            "type": "door",
        },
    ],
    "windows": [
        {"room": "Living Room", "count": 2},
        {"room": "Kitchen", "count": 1},
        {"room": "Bedroom 1", "count": 1},
        {"room": "Bathroom", "count": 1},
    ],
    "confidence": "high",
    "description": (
        "A compact 80 m² single-floor apartment with an open-plan "
        "living/kitchen area, one bedroom, a bathroom, and a central hallway."
    ),
}


class _MockVisionModel(ChatModelBase):
    """Mock model that returns a generate_response tool call with fake data."""

    def __init__(self) -> None:
        """Initialise with no real API credentials."""
        super().__init__("mock_vision_model", stream=False)
        self._call_count = 0

    async def __call__(
        self,
        _messages: list[dict],
        tools: list[dict] | None = None,
        **_kwargs: Any,
    ) -> ChatResponse:
        """Return a generate_response tool call on first invocation."""
        self._call_count += 1
        return ChatResponse(
            content=[
                ToolUseBlock(
                    type="tool_use",
                    id="mock_tool_call_id",
                    name="generate_response",
                    input=_FAKE_FLOOR_PLAN,
                ),
            ],
        )


class FloorPlanExtractorTest(IsolatedAsyncioTestCase):
    """Tests for :class:`FloorPlanExtractor`."""

    def setUp(self) -> None:
        """Create the extractor with a mock model."""
        self.extractor = FloorPlanExtractor(
            model=_MockVisionModel(),
            formatter=DashScopeChatFormatter(),
        )

    async def test_extract_from_url(self) -> None:
        """Extractor should return a valid FloorPlan when given a URL."""
        result = await self.extractor.extract(
            ["https://example.com/plan.png"],
        )
        self.assertIsInstance(result, FloorPlan)
        self.assertEqual(result.num_floors, 1)
        self.assertAlmostEqual(result.total_area_sqm, 80.0)  # type: ignore[arg-type]
        self.assertEqual(len(result.rooms), 5)
        self.assertEqual(result.rooms[0].name, "Living Room")
        self.assertEqual(result.confidence, "high")

    async def test_extract_from_file(self) -> None:
        """Extractor should load a local file as base64 and return FloorPlan."""
        # Create a minimal 1×1 PNG in a temp file
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(png_bytes)
            tmp_path = tmp.name

        result = await self.extractor.extract([tmp_path])
        self.assertIsInstance(result, FloorPlan)
        self.assertEqual(len(result.rooms), 5)

    async def test_extract_multiple_images(self) -> None:
        """Extractor should accept several images without raising errors."""
        result = await self.extractor.extract(
            [
                "https://example.com/ground_floor.png",
                "https://example.com/first_floor.png",
            ],
        )
        self.assertIsInstance(result, FloorPlan)

    async def test_extract_with_extra_instructions(self) -> None:
        """Extra instructions should be included without breaking extraction."""
        result = await self.extractor.extract(
            ["https://example.com/plan.png"],
            extra_instructions="Focus on the ground floor only.",
        )
        self.assertIsInstance(result, FloorPlan)

    async def test_extract_raises_on_empty_images(self) -> None:
        """extract() should raise ValueError when no images are given."""
        with self.assertRaises(ValueError):
            await self.extractor.extract([])

    async def test_floor_plan_model_validation(self) -> None:
        """FloorPlan Pydantic model should validate the fake payload."""
        plan = FloorPlan.model_validate(_FAKE_FLOOR_PLAN)
        self.assertEqual(plan.num_floors, 1)
        self.assertEqual(len(plan.openings), 5)
        self.assertEqual(len(plan.windows), 4)
        self.assertEqual(plan.rooms[3].type, "bedroom")

    def test_load_image_block_url(self) -> None:
        """_load_image_block should return a URLSource block for HTTP URLs."""
        from _extractor import _load_image_block

        block = _load_image_block("https://example.com/image.jpg")
        # ImageBlock is a TypedDict – access via keys
        self.assertEqual(block["type"], "image")
        self.assertEqual(block["source"]["type"], "url")
        self.assertEqual(block["source"]["url"], "https://example.com/image.jpg")

    def test_load_image_block_file(self) -> None:
        """_load_image_block should return a Base64Source block for files."""
        from _extractor import _load_image_block

        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(png_bytes)
            tmp_path = tmp.name

        block = _load_image_block(tmp_path)
        self.assertEqual(block["type"], "image")
        self.assertEqual(block["source"]["type"], "base64")
        self.assertEqual(block["source"]["media_type"], "image/png")

    async def test_result_rooms_have_connections(self) -> None:
        """Rooms should carry their adjacency list."""
        result = await self.extractor.extract(
            ["https://example.com/plan.png"],
        )
        living = next(r for r in result.rooms if r.name == "Living Room")
        self.assertIn("Kitchen", living.connected_to)
        self.assertIn("Hallway", living.connected_to)
