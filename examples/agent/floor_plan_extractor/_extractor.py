# -*- coding: utf-8 -*-
"""Floor plan extractor using AgentScope vision models."""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Sequence

from agentscope.formatter import FormatterBase
from agentscope.memory import InMemoryMemory
from agentscope.message import (
    Base64Source,
    ImageBlock,
    Msg,
    TextBlock,
    URLSource,
)
from agentscope.model import ChatModelBase
from agentscope.tool import Toolkit

from _models import FloorPlan

_SYS_PROMPT = """\
You are an expert architectural analyst specialised in floor plan extraction.

Given one or more images (floor plan drawings, photographs of plans, or \
interior/exterior room photos), your task is to produce a structured \
description of the floor plan with the following details:

- All identifiable rooms with their type, estimated dimensions, and \
  connections to adjacent rooms
- Doors, archways, and other openings between rooms or to the exterior
- Window count per room where visible
- Total estimated area (if inferable)
- An overall confidence rating and a brief natural-language summary

Guidelines:
- If the image is a clear architectural drawing, extract exact room labels \
  and dimensions when visible.
- If the image is a photograph of a plan, make your best estimate.
- If the images are interior room photos with no explicit plan, infer the \
  layout from visible spatial cues (doorways, corridors, etc.) and set \
  confidence to 'low'.
- Use metric units (metres, square metres).
- When dimensions are truly impossible to determine, set the corresponding \
  field to null rather than guessing.
"""


def _load_image_block(source: str | Path) -> ImageBlock:
    """Convert a file path or URL string to an :class:`ImageBlock`.

    Args:
        source (`str | Path`):
            A local file path or a remote URL string.

    Returns:
        `ImageBlock`:
            The corresponding image content block.
    """
    path = Path(source)
    if path.exists():
        mime, _ = mimetypes.guess_type(str(path))
        if mime is None:
            mime = "image/jpeg"
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return ImageBlock(
            type="image",
            source=Base64Source(
                type="base64",
                media_type=mime,  # type: ignore[arg-type]
                data=data,
            ),
        )
    # Treat as URL
    return ImageBlock(
        type="image",
        source=URLSource(type="url", url=str(source)),
    )


class FloorPlanExtractor:
    """Extract a structured floor plan from one or more images.

    The extractor wraps an AgentScope :class:`~agentscope.agent.ReActAgent`
    configured with a vision-capable model.  Images can be provided as local
    file paths or remote URLs.

    Args:
        model (`ChatModelBase`):
            A vision-capable chat model (e.g.
            :class:`~agentscope.model.OpenAIChatModel` with ``gpt-4o``,
            :class:`~agentscope.model.AnthropicChatModel` with
            ``claude-opus-4-6``, or
            :class:`~agentscope.model.DashScopeChatModel` with
            ``qwen3-vl-plus``).
        formatter (`FormatterBase`):
            The message formatter matching the chosen model provider.

    Example:

    .. code-block:: python

        import asyncio
        import os
        from agentscope.model import OpenAIChatModel
        from agentscope.formatter import OpenAIChatFormatter
        from floor_plan_extractor import FloorPlanExtractor

        extractor = FloorPlanExtractor(
            model=OpenAIChatModel(
                model_name="gpt-4o",
                api_key=os.environ["OPENAI_API_KEY"],
            ),
            formatter=OpenAIChatFormatter(),
        )

        floor_plan = asyncio.run(
            extractor.extract(["floor_plan.png"])
        )
        print(floor_plan.model_dump_json(indent=2))
    """

    def __init__(
        self,
        model: ChatModelBase,
        formatter: FormatterBase,
    ) -> None:
        """Initialise the extractor with a vision model and formatter."""
        # Import here to avoid circular dependency issues in tests
        from agentscope.agent import ReActAgent

        self._agent = ReActAgent(
            name="FloorPlanExtractor",
            sys_prompt=_SYS_PROMPT,
            model=model,
            formatter=formatter,
            toolkit=Toolkit(),
            memory=InMemoryMemory(),
        )

    async def extract(
        self,
        images: Sequence[str | Path],
        extra_instructions: str | None = None,
    ) -> FloorPlan:
        """Extract a :class:`FloorPlan` from the provided images.

        Args:
            images (`Sequence[str | Path]`):
                One or more image sources.  Each entry can be:

                - A local file path (``str`` or :class:`~pathlib.Path`).
                - A remote URL string (``http://`` or ``https://``).

            extra_instructions (`str | None`, optional):
                Optional free-text instructions appended to the user message
                (e.g. ``"Focus only on the ground floor."``).
                Defaults to None.

        Returns:
            `FloorPlan`:
                Structured floor plan data extracted from the images.

        Raises:
            ValueError:
                If ``images`` is empty.
        """
        if not images:
            raise ValueError("At least one image must be provided.")

        content: list[TextBlock | ImageBlock] = []

        prompt_text = (
            "Please extract the floor plan from the following "
            f"{'image' if len(images) == 1 else f'{len(images)} images'}."
        )
        if extra_instructions:
            prompt_text += f"\n\nAdditional instructions: {extra_instructions}"

        content.append(TextBlock(type="text", text=prompt_text))

        for img in images:
            content.append(_load_image_block(img))

        msg = Msg(name="user", content=content, role="user")
        result = await self._agent(msg, structured_model=FloorPlan)

        return FloorPlan.model_validate(result.metadata)
