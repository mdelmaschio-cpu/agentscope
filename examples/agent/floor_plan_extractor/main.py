# -*- coding: utf-8 -*-
"""Floor plan extraction example.

This script demonstrates how to use :class:`FloorPlanExtractor` to analyse
one or more images and obtain a structured :class:`FloorPlan` object.

Usage
-----
Set the appropriate environment variable for your chosen provider, then run::

    # OpenAI
    export OPENAI_API_KEY=sk-...
    python main.py --provider openai path/to/floor_plan.png

    # Anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py --provider anthropic path/to/floor_plan.png

    # DashScope (Alibaba Cloud)
    export DASHSCOPE_API_KEY=sk-...
    python main.py --provider dashscope path/to/floor_plan.png

    # Multiple images
    python main.py path/to/ground_floor.png path/to/first_floor.png

    # Remote URL
    python main.py https://example.com/plan.jpg
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Literal

from agentscope.formatter import (
    AnthropicChatFormatter,
    DashScopeChatFormatter,
    OpenAIChatFormatter,
)
from agentscope.model import (
    AnthropicChatModel,
    DashScopeChatModel,
    OpenAIChatModel,
)

from _extractor import FloorPlanExtractor

Provider = Literal["openai", "anthropic", "dashscope"]

_DEFAULTS: dict[str, dict[str, object]] = {
    "openai": {"model_name": "gpt-4o", "env": "OPENAI_API_KEY"},
    "anthropic": {
        "model_name": "claude-opus-4-6",
        "env": "ANTHROPIC_API_KEY",
    },
    "dashscope": {"model_name": "qwen3-vl-plus", "env": "DASHSCOPE_API_KEY"},
}


def _build_extractor(provider: Provider, model_name: str) -> FloorPlanExtractor:
    """Instantiate a :class:`FloorPlanExtractor` for the chosen provider.

    Args:
        provider (`Provider`):
            One of ``"openai"``, ``"anthropic"``, or ``"dashscope"``.
        model_name (`str`):
            Model identifier for the chosen provider.

    Returns:
        `FloorPlanExtractor`:
            Configured extractor instance.
    """
    env_key = str(_DEFAULTS[provider]["env"])
    api_key = os.environ.get(env_key)
    if api_key is None:
        print(
            f"[warn] Environment variable {env_key!r} is not set. "
            "The request may fail unless the SDK finds credentials elsewhere.",
            file=sys.stderr,
        )

    if provider == "openai":
        return FloorPlanExtractor(
            model=OpenAIChatModel(model_name=model_name, api_key=api_key),
            formatter=OpenAIChatFormatter(),
        )
    if provider == "anthropic":
        return FloorPlanExtractor(
            model=AnthropicChatModel(model_name=model_name, api_key=api_key),
            formatter=AnthropicChatFormatter(),
        )
    # dashscope
    return FloorPlanExtractor(
        model=DashScopeChatModel(model_name=model_name, api_key=api_key),
        formatter=DashScopeChatFormatter(),
    )


async def main() -> None:
    """Entry point for the floor plan extraction demo."""
    parser = argparse.ArgumentParser(
        description="Extract a structured floor plan from images.",
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="One or more image paths or URLs.",
    )
    parser.add_argument(
        "--provider",
        choices=list(_DEFAULTS.keys()),
        default="openai",
        help="LLM provider to use (default: openai).",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        default=None,
        help=(
            "Model name override. "
            "Defaults: openai=gpt-4o, anthropic=claude-opus-4-6, "
            "dashscope=qwen3-vl-plus."
        ),
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Optional extra instructions for the model.",
    )
    args = parser.parse_args()

    provider: Provider = args.provider  # type: ignore[assignment]
    model_name: str = args.model_name or str(_DEFAULTS[provider]["model_name"])

    extractor = _build_extractor(provider, model_name)

    print(
        f"Extracting floor plan from {len(args.images)} image(s) "
        f"using {provider}/{model_name} …\n",
    )

    floor_plan = await extractor.extract(
        images=args.images,
        extra_instructions=args.instructions,
    )

    print("=== Extracted Floor Plan ===")
    print(json.dumps(floor_plan.model_dump(), indent=2, ensure_ascii=False))


asyncio.run(main())
