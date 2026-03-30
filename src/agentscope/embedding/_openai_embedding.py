# -*- coding: utf-8 -*-
"""The OpenAI text embedding model class."""
from datetime import datetime
from typing import Any, List

from ._embedding_response import EmbeddingResponse
from ._embedding_usage import EmbeddingUsage
from ._cache_base import EmbeddingCacheBase
from ._embedding_base import EmbeddingModelBase
from ..message import TextBlock


class OpenAITextEmbedding(EmbeddingModelBase):
    """OpenAI text embedding model class."""

    supported_modalities: list[str] = ["text"]
    """This class only supports text input."""

    max_batch_size: int = 2048
    """Maximum number of inputs per API request (OpenAI limit)."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        dimensions: int = 1024,
        embedding_cache: EmbeddingCacheBase | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI text embedding model class.

        Args:
            api_key (`str`):
                The OpenAI API key.
            model_name (`str`):
                The name of the embedding model.
            dimensions (`int`, defaults to 1024):
                The dimension of the embedding vector.
            embedding_cache (`EmbeddingCacheBase | None`, defaults to `None`):
                The embedding cache class instance, used to cache the
                embedding results to avoid repeated API calls.
        """
        import openai

        super().__init__(model_name, dimensions)

        self.client = openai.AsyncClient(api_key=api_key, **kwargs)
        self.embedding_cache = embedding_cache

    async def __call__(
        self,
        text: List[str | TextBlock],
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Call the OpenAI embedding API.

        Inputs exceeding ``max_batch_size`` are automatically split into
        multiple requests and the results are merged.

        Args:
            text (`List[str | TextBlock]`):
                The input text to be embedded. It can be a list of strings
                or ``TextBlock`` dicts.
        """
        gather_text = []
        for _ in text:
            if isinstance(_, dict) and "text" in _:
                gather_text.append(_["text"])
            elif isinstance(_, str):
                gather_text.append(_)
            else:
                raise ValueError(
                    "Input text must be a list of strings or TextBlock dicts.",
                )

        all_embeddings: List[Any] = []
        total_tokens = 0
        total_time = 0.0

        for batch_start in range(0, len(gather_text), self.max_batch_size):
            batch = gather_text[batch_start : batch_start + self.max_batch_size]
            batch_kwargs = {
                "input": batch,
                "model": self.model_name,
                "dimensions": self.dimensions,
                "encoding_format": "float",
                **kwargs,
            }

            if self.embedding_cache:
                cached_embeddings = await self.embedding_cache.retrieve(
                    identifier=batch_kwargs,
                )
                if cached_embeddings:
                    all_embeddings.extend(cached_embeddings)
                    continue

            start_time = datetime.now()
            response = await self.client.embeddings.create(**batch_kwargs)
            total_time += (datetime.now() - start_time).total_seconds()
            total_tokens += response.usage.total_tokens

            batch_embeddings = [_.embedding for _ in response.data]

            if self.embedding_cache:
                await self.embedding_cache.store(
                    identifier=batch_kwargs,
                    embeddings=batch_embeddings,
                )

            all_embeddings.extend(batch_embeddings)

        return EmbeddingResponse(
            embeddings=all_embeddings,
            usage=EmbeddingUsage(
                tokens=total_tokens,
                time=total_time,
            ),
        )
