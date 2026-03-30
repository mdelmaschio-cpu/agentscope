# -*- coding: utf-8 -*-
"""The gemini text embedding model class."""
from datetime import datetime
from typing import Any, List

from ._embedding_response import EmbeddingResponse
from ._embedding_usage import EmbeddingUsage
from ._cache_base import EmbeddingCacheBase
from ._embedding_base import EmbeddingModelBase
from ..message import TextBlock


class GeminiTextEmbedding(EmbeddingModelBase):
    """The Gemini text embedding model."""

    supported_modalities: list[str] = ["text"]
    """This class only supports text input."""

    max_batch_size: int = 100
    """Maximum number of inputs per API request (Gemini limit)."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        dimensions: int = 3072,
        embedding_cache: EmbeddingCacheBase | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Gemini text embedding model class.

        Args:
            api_key (`str`):
                The Gemini API key.
            model_name (`str`):
                The name of the embedding model.
            dimensions (`int`, defaults to 3072):
                The dimension of the embedding vector, refer to the
                `official documentation
                <https://ai.google.dev/gemini-api/docs/embeddings?hl=zh-cn#control-embedding-size>`_
                for more details.
            embedding_cache (`EmbeddingCacheBase | None`, defaults to `None`):
                The embedding cache class instance, used to cache the
                embedding results to avoid repeated API calls.
        """
        from google import genai

        super().__init__(model_name, dimensions)

        self.client = genai.Client(api_key=api_key, **kwargs)
        self.embedding_cache = embedding_cache

    async def __call__(
        self,
        text: List[str | TextBlock],
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """The Gemini embedding API call.

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
        total_time = 0.0

        for batch_start in range(0, len(gather_text), self.max_batch_size):
            batch = gather_text[batch_start : batch_start + self.max_batch_size]
            batch_kwargs = {
                "model": self.model_name,
                "contents": batch,
                "config": kwargs,
            }

            if self.embedding_cache:
                cached_embeddings = await self.embedding_cache.retrieve(
                    identifier=batch_kwargs,
                )
                if cached_embeddings:
                    all_embeddings.extend(cached_embeddings)
                    continue

            start_time = datetime.now()
            response = self.client.models.embed_content(**batch_kwargs)
            total_time += (datetime.now() - start_time).total_seconds()

            batch_embeddings = [_.values for _ in response.embeddings]

            if self.embedding_cache:
                await self.embedding_cache.store(
                    identifier=batch_kwargs,
                    embeddings=batch_embeddings,
                )

            all_embeddings.extend(batch_embeddings)

        return EmbeddingResponse(
            embeddings=all_embeddings,
            usage=EmbeddingUsage(
                time=total_time,
            ),
        )
