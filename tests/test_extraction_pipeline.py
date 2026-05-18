"""
Unit tests for the ContentExtractionPipeline.

Tests cover:
  - Tier selection and fallback behaviour
  - Playwright unavailability (graceful degradation)
  - Trafilatura / Readability returning None (escalation)
  - Docling failure (fallback to BeautifulSoup)
  - LLM cognitive refinement (triggered and skipped)
  - Binary-document extraction via extract_from_bytes
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.mcp_server.extraction.pipeline import (
    ContentExtractionPipeline,
    ExtractionResult,
    _MIN_WORD_COUNT,
    _LLM_TRIGGER_WORD_COUNT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _words(n: int) -> str:
    """Return a string of *n* space-separated 'word' tokens."""
    return " ".join(["word"] * n)


RICH_HTML = f"<html><body><p>{_words(300)}</p></body></html>"
SHORT_HTML = "<html><body><h1>Hi</h1><p>Short.</p></body></html>"


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------

class TestExtractionResult:
    def test_word_count_computed(self):
        r = ExtractionResult(content="hello world foo", method="test")
        assert r.word_count == 3

    def test_empty_content(self):
        r = ExtractionResult(content="", method="test")
        assert r.word_count == 0


# ---------------------------------------------------------------------------
# Playwright tier
# ---------------------------------------------------------------------------

class TestPlaywrightTier:
    @pytest.mark.asyncio
    async def test_playwright_unavailable_falls_back_to_static(self):
        """When Playwright cannot be imported, use static HTML + next tier."""
        pipeline = ContentExtractionPipeline()

        # Playwright fails → _render_with_playwright returns None
        with patch.object(pipeline, "_render_with_playwright", new=AsyncMock(return_value=None)):
            # Trafilatura returns rich content from static HTML
            with patch.object(
                ContentExtractionPipeline,
                "_extract_trafilatura",
                return_value=_words(_MIN_WORD_COUNT + 10),
            ):
                result = await pipeline.extract_from_html(
                    html=RICH_HTML, url="https://example.com", use_playwright=True
                )

        assert "static" in result.method
        assert result.word_count >= _MIN_WORD_COUNT

    @pytest.mark.asyncio
    async def test_playwright_success_sets_prefix(self):
        """When Playwright succeeds, method prefix should be 'playwright'."""
        pipeline = ContentExtractionPipeline()
        rendered_html = f"<html><body><p>{_words(300)}</p></body></html>"

        with patch.object(
            pipeline, "_render_with_playwright", new=AsyncMock(return_value=rendered_html)
        ):
            with patch.object(
                ContentExtractionPipeline,
                "_extract_trafilatura",
                return_value=_words(_MIN_WORD_COUNT + 10),
            ):
                result = await pipeline.extract_from_html(
                    html="<html><body></body></html>",
                    url="https://example.com",
                    use_playwright=True,
                )

        assert result.method.startswith("playwright")

    @pytest.mark.asyncio
    async def test_playwright_disabled_skips_tier(self):
        """use_playwright=False should never call _render_with_playwright."""
        pipeline = ContentExtractionPipeline()
        mock_render = AsyncMock(return_value=None)

        with patch.object(pipeline, "_render_with_playwright", mock_render):
            with patch.object(
                ContentExtractionPipeline,
                "_extract_trafilatura",
                return_value=_words(_MIN_WORD_COUNT + 10),
            ):
                result = await pipeline.extract_from_html(
                    html=RICH_HTML, url="https://example.com", use_playwright=False
                )

        mock_render.assert_not_called()
        assert result.method.startswith("static")


# ---------------------------------------------------------------------------
# Heuristic tier escalation
# ---------------------------------------------------------------------------

class TestHeuristicTier:
    @pytest.mark.asyncio
    async def test_trafilatura_success_stops_escalation(self):
        """Sufficient Trafilatura output should prevent Readability from running."""
        pipeline = ContentExtractionPipeline()
        readability_mock = MagicMock(return_value=_words(200))

        with patch.object(pipeline, "_render_with_playwright", new=AsyncMock(return_value=None)):
            with patch.object(
                ContentExtractionPipeline,
                "_extract_trafilatura",
                return_value=_words(250),  # above _RICH_WORD_COUNT
            ):
                with patch.object(
                    ContentExtractionPipeline, "_extract_readability", readability_mock
                ):
                    result = await pipeline.extract_from_html(
                        html=RICH_HTML, url="https://example.com", use_playwright=False
                    )

        readability_mock.assert_not_called()
        assert "trafilatura" in result.method

    @pytest.mark.asyncio
    async def test_trafilatura_none_escalates_to_readability(self):
        """None from Trafilatura should escalate to Readability-lxml."""
        pipeline = ContentExtractionPipeline()

        with patch.object(pipeline, "_render_with_playwright", new=AsyncMock(return_value=None)):
            with patch.object(
                ContentExtractionPipeline, "_extract_trafilatura", return_value=None
            ):
                with patch.object(
                    ContentExtractionPipeline,
                    "_extract_readability",
                    return_value=_words(250),
                ):
                    result = await pipeline.extract_from_html(
                        html=RICH_HTML, url="https://example.com", use_playwright=False
                    )

        assert "readability" in result.method


# ---------------------------------------------------------------------------
# Docling / BeautifulSoup fallback
# ---------------------------------------------------------------------------

class TestDoclingAndBsFallback:
    @pytest.mark.asyncio
    async def test_docling_used_when_heuristics_fail(self):
        """When heuristic tiers return too little, Docling should be tried."""
        pipeline = ContentExtractionPipeline()

        with patch.object(pipeline, "_render_with_playwright", new=AsyncMock(return_value=None)):
            with patch.object(
                ContentExtractionPipeline, "_extract_trafilatura", return_value=None
            ):
                with patch.object(
                    ContentExtractionPipeline, "_extract_readability", return_value=None
                ):
                    with patch.object(
                        ContentExtractionPipeline,
                        "_extract_docling_html",
                        new=AsyncMock(return_value=_words(250)),
                    ):
                        result = await pipeline.extract_from_html(
                            html=RICH_HTML, url="https://example.com", use_playwright=False
                        )

        assert "docling" in result.method

    @pytest.mark.asyncio
    async def test_beautifulsoup_final_fallback(self):
        """BeautifulSoup is always reached when all other tiers yield nothing."""
        pipeline = ContentExtractionPipeline()
        bs_content = _words(30)  # below rich threshold but above zero

        with patch.object(pipeline, "_render_with_playwright", new=AsyncMock(return_value=None)):
            with patch.object(
                ContentExtractionPipeline, "_extract_trafilatura", return_value=None
            ):
                with patch.object(
                    ContentExtractionPipeline, "_extract_readability", return_value=None
                ):
                    with patch.object(
                        ContentExtractionPipeline,
                        "_extract_docling_html",
                        new=AsyncMock(return_value=None),
                    ):
                        with patch.object(
                            ContentExtractionPipeline,
                            "_extract_beautifulsoup",
                            new=AsyncMock(return_value=bs_content),
                        ):
                            result = await pipeline.extract_from_html(
                                html=SHORT_HTML,
                                url="https://example.com",
                                use_playwright=False,
                            )

        assert "beautifulsoup" in result.method
        assert result.content == bs_content


# ---------------------------------------------------------------------------
# LLM cognitive refinement
# ---------------------------------------------------------------------------

class TestLLMRefinement:
    @pytest.mark.asyncio
    async def test_llm_refinement_triggered_when_content_is_poor(self):
        """LLM refinement should be called when content is below the trigger threshold."""
        pipeline = ContentExtractionPipeline()
        poor_content = _words(_LLM_TRIGGER_WORD_COUNT - 10)  # below threshold
        refined_content = _words(_LLM_TRIGGER_WORD_COUNT + 20)  # LLM improves it

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=refined_content)

        with patch.object(pipeline, "_render_with_playwright", new=AsyncMock(return_value=None)):
            with patch.object(
                ContentExtractionPipeline, "_extract_trafilatura", return_value=None
            ):
                with patch.object(
                    ContentExtractionPipeline, "_extract_readability", return_value=None
                ):
                    with patch.object(
                        ContentExtractionPipeline,
                        "_extract_docling_html",
                        new=AsyncMock(return_value=None),
                    ):
                        with patch.object(
                            ContentExtractionPipeline,
                            "_extract_beautifulsoup",
                            new=AsyncMock(return_value=poor_content),
                        ):
                            result = await pipeline.extract_from_html(
                                html=SHORT_HTML,
                                url="https://example.com",
                                use_playwright=False,
                                use_llm_refinement=True,
                                llm_manager=mock_llm,
                            )

        assert "llm" in result.method
        assert result.content == refined_content
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_refinement_skipped_when_content_is_rich(self):
        """LLM refinement must NOT be called when content already exceeds the threshold."""
        pipeline = ContentExtractionPipeline()
        rich_content = _words(_LLM_TRIGGER_WORD_COUNT + 50)

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value="should not be called")

        with patch.object(pipeline, "_render_with_playwright", new=AsyncMock(return_value=None)):
            with patch.object(
                ContentExtractionPipeline,
                "_extract_trafilatura",
                return_value=rich_content,
            ):
                result = await pipeline.extract_from_html(
                    html=RICH_HTML,
                    url="https://example.com",
                    use_playwright=False,
                    use_llm_refinement=True,
                    llm_manager=mock_llm,
                )

        mock_llm.complete.assert_not_called()
        assert "llm" not in result.method

    @pytest.mark.asyncio
    async def test_llm_refinement_skipped_when_disabled(self):
        """use_llm_refinement=False must never call the LLM even for poor content."""
        pipeline = ContentExtractionPipeline()
        poor_content = _words(10)

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value="should not be called")

        with patch.object(pipeline, "_render_with_playwright", new=AsyncMock(return_value=None)):
            with patch.object(
                ContentExtractionPipeline, "_extract_trafilatura", return_value=None
            ):
                with patch.object(
                    ContentExtractionPipeline, "_extract_readability", return_value=None
                ):
                    with patch.object(
                        ContentExtractionPipeline,
                        "_extract_docling_html",
                        new=AsyncMock(return_value=None),
                    ):
                        with patch.object(
                            ContentExtractionPipeline,
                            "_extract_beautifulsoup",
                            new=AsyncMock(return_value=poor_content),
                        ):
                            result = await pipeline.extract_from_html(
                                html=SHORT_HTML,
                                url="https://example.com",
                                use_playwright=False,
                                use_llm_refinement=False,
                                llm_manager=mock_llm,
                            )

        mock_llm.complete.assert_not_called()
        assert "llm" not in result.method

    @pytest.mark.asyncio
    async def test_llm_failure_keeps_original_content(self):
        """When LLM refinement raises an exception, original content is preserved."""
        pipeline = ContentExtractionPipeline()
        poor_content = _words(_LLM_TRIGGER_WORD_COUNT - 5)

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        with patch.object(pipeline, "_render_with_playwright", new=AsyncMock(return_value=None)):
            with patch.object(
                ContentExtractionPipeline, "_extract_trafilatura", return_value=None
            ):
                with patch.object(
                    ContentExtractionPipeline, "_extract_readability", return_value=None
                ):
                    with patch.object(
                        ContentExtractionPipeline,
                        "_extract_docling_html",
                        new=AsyncMock(return_value=None),
                    ):
                        with patch.object(
                            ContentExtractionPipeline,
                            "_extract_beautifulsoup",
                            new=AsyncMock(return_value=poor_content),
                        ):
                            result = await pipeline.extract_from_html(
                                html=SHORT_HTML,
                                url="https://example.com",
                                use_playwright=False,
                                use_llm_refinement=True,
                                llm_manager=mock_llm,
                            )

        assert result.content == poor_content
        assert "llm" not in result.method


# ---------------------------------------------------------------------------
# Binary document extraction
# ---------------------------------------------------------------------------

class TestExtractFromBytes:
    @pytest.mark.asyncio
    async def test_docling_success(self):
        pipeline = ContentExtractionPipeline()
        rich_content = _words(200)

        with patch(
            "mcp_server.llm.parser.parse_with_docling",
            new=AsyncMock(return_value=rich_content),
        ):
            result = await pipeline.extract_from_bytes(b"%PDF-1.4...", ".pdf")

        assert result.method == "docling"
        assert result.content == rich_content

    @pytest.mark.asyncio
    async def test_docling_failure_falls_back_to_beautifulsoup(self):
        pipeline = ContentExtractionPipeline()

        with patch(
            "mcp_server.llm.parser.parse_with_docling",
            new=AsyncMock(return_value=None),
        ):
            with patch(
                "mcp_server.llm.parser.parse_html_with_beautifulsoup",
                new=AsyncMock(return_value="fallback content"),
            ):
                result = await pipeline.extract_from_bytes(b"<html>...</html>", ".html")

        assert result.method == "beautifulsoup"
        assert result.content == "fallback content"
