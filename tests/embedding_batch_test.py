# -*- coding: utf-8 -*-
"""Unit tests for batch size handling in embedding models."""
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from agentscope.embedding._openai_embedding import OpenAITextEmbedding
from agentscope.embedding._gemini_embedding import GeminiTextEmbedding


def _openai_mock() -> MagicMock:
    """Return a minimal mock of the openai module."""
    mock_openai = MagicMock()
    mock_client = MagicMock()
    mock_openai.AsyncClient.return_value = mock_client
    return mock_openai


def _google_mock() -> MagicMock:
    """Return a minimal mock of the google.genai module."""
    mock_google = MagicMock()
    mock_client = MagicMock()
    mock_google.genai.Client.return_value = mock_client
    return mock_google


def _openai_response(n: int) -> MagicMock:
    """Build a fake openai embeddings.create response for *n* inputs."""
    resp = MagicMock()
    resp.data = [MagicMock(embedding=[float(i)] * 4) for i in range(n)]
    resp.usage.total_tokens = n * 10
    return resp


def _gemini_response(n: int) -> MagicMock:
    """Build a fake Gemini embed_content response for *n* inputs."""
    resp = MagicMock()
    resp.embeddings = [MagicMock(values=[float(i)] * 4) for i in range(n)]
    return resp


class TestOpenAIEmbeddingBatch(IsolatedAsyncioTestCase):
    """Tests for batch size handling in OpenAITextEmbedding."""

    async def test_single_batch(self) -> None:
        """Inputs within max_batch_size should produce exactly one API call."""
        mock_openai = _openai_mock()
        mock_client = mock_openai.AsyncClient.return_value
        mock_client.embeddings.create = AsyncMock(
            return_value=_openai_response(2),
        )

        with patch.dict("sys.modules", {"openai": mock_openai}):
            model = OpenAITextEmbedding(
                api_key="test",
                model_name="text-embedding-3-small",
                dimensions=4,
            )
            result = await model(["hello", "world"])

        self.assertEqual(len(result.embeddings), 2)
        self.assertEqual(mock_client.embeddings.create.call_count, 1)
        self.assertEqual(result.usage.tokens, 20)

    async def test_multiple_batches(self) -> None:
        """Inputs exceeding max_batch_size are split across multiple calls."""
        mock_openai = _openai_mock()
        mock_client = mock_openai.AsyncClient.return_value

        call_count = 0

        async def fake_create(**kwargs: object) -> MagicMock:
            nonlocal call_count
            batch = kwargs["input"]
            call_count += 1
            return _openai_response(len(batch))  # type: ignore[arg-type]

        mock_client.embeddings.create = fake_create

        original_batch = OpenAITextEmbedding.max_batch_size
        OpenAITextEmbedding.max_batch_size = 3  # force 2 batches for 5 inputs

        try:
            with patch.dict("sys.modules", {"openai": mock_openai}):
                model = OpenAITextEmbedding(
                    api_key="test",
                    model_name="text-embedding-3-small",
                    dimensions=4,
                )
                result = await model(["a", "b", "c", "d", "e"])
        finally:
            OpenAITextEmbedding.max_batch_size = original_batch

        self.assertEqual(len(result.embeddings), 5)
        self.assertEqual(call_count, 2)
        # batch 1: 3 items × 10 tokens + batch 2: 2 items × 10 tokens = 50
        self.assertEqual(result.usage.tokens, 50)

    async def test_invalid_input_raises(self) -> None:
        """A non-string, non-TextBlock element should raise ValueError."""
        mock_openai = _openai_mock()

        with patch.dict("sys.modules", {"openai": mock_openai}):
            model = OpenAITextEmbedding(
                api_key="test",
                model_name="text-embedding-3-small",
                dimensions=4,
            )
            with self.assertRaises(ValueError):
                await model([123])  # type: ignore[list-item]


class TestGeminiEmbeddingBatch(IsolatedAsyncioTestCase):
    """Tests for batch size handling in GeminiTextEmbedding."""

    async def test_single_batch(self) -> None:
        """Inputs within max_batch_size should produce exactly one API call."""
        mock_google = _google_mock()
        mock_client = mock_google.genai.Client.return_value
        mock_client.models.embed_content = MagicMock(
            return_value=_gemini_response(2),
        )

        with patch.dict(
            "sys.modules",
            {"google": mock_google, "google.genai": mock_google.genai},
        ):
            model = GeminiTextEmbedding(
                api_key="test",
                model_name="models/text-embedding-004",
                dimensions=4,
            )
            result = await model(["hello", "world"])

        self.assertEqual(len(result.embeddings), 2)
        self.assertEqual(mock_client.models.embed_content.call_count, 1)

    async def test_multiple_batches(self) -> None:
        """Inputs exceeding max_batch_size are split across multiple calls."""
        mock_google = _google_mock()
        mock_client = mock_google.genai.Client.return_value

        call_count = 0

        def fake_embed(**kwargs: object) -> MagicMock:
            nonlocal call_count
            batch = kwargs["contents"]
            call_count += 1
            return _gemini_response(len(batch))  # type: ignore[arg-type]

        mock_client.models.embed_content = fake_embed

        original_batch = GeminiTextEmbedding.max_batch_size
        GeminiTextEmbedding.max_batch_size = 2  # force 3 batches for 5 inputs

        try:
            with patch.dict(
                "sys.modules",
                {"google": mock_google, "google.genai": mock_google.genai},
            ):
                model = GeminiTextEmbedding(
                    api_key="test",
                    model_name="models/text-embedding-004",
                    dimensions=4,
                )
                result = await model(["a", "b", "c", "d", "e"])
        finally:
            GeminiTextEmbedding.max_batch_size = original_batch

        self.assertEqual(len(result.embeddings), 5)
        self.assertEqual(call_count, 3)

    async def test_invalid_input_raises(self) -> None:
        """A non-string, non-TextBlock element should raise ValueError."""
        mock_google = _google_mock()

        with patch.dict(
            "sys.modules",
            {"google": mock_google, "google.genai": mock_google.genai},
        ):
            model = GeminiTextEmbedding(
                api_key="test",
                model_name="models/text-embedding-004",
                dimensions=4,
            )
            with self.assertRaises(ValueError):
                await model([42])  # type: ignore[list-item]
