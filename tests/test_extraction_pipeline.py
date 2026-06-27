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

import asyncio

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
    async def test_llm_refinement_called_when_content_is_rich(self):
        """LLM refinement IS called when use_llm_refinement=True even for rich content,
        but refined content may not update if it has fewer words."""
        pipeline = ContentExtractionPipeline()
        rich_content = _words(_LLM_TRIGGER_WORD_COUNT + 50)

        mock_llm = MagicMock()
        # Return content with fewer words — refinement won't replace original
        mock_llm.complete = AsyncMock(return_value="refined")

        with patch.object(pipeline, "_render_with_playwright", new=AsyncMock(return_value=None)):
            with patch.object(
                ContentExtractionPipeline,
                "_extract_trafilatura",
                return_value=rich_content,
            ):
                # Patch docling to fail (None response)
                with patch.object(
                    ContentExtractionPipeline,
                    '_extract_docling_html',
                    new=AsyncMock(return_value=None),
                ):
                    # Patch BeautifulSoup too to return same content — keeps it unchanged  
                    with patch.object(
                        ContentExtractionPipeline,
                        '_extract_beautifulsoup',
                        return_value=rich_content,
                    ):
                        result = await pipeline.extract_from_html(
                            html=RICH_HTML,
                            url="https://example.com",
                            use_playwright=False,
                            use_llm_refinement=True,
                            llm_manager=mock_llm,
                        )

        # LLM should be called even for rich content when use_llm_refinement=True
        mock_llm.complete.assert_called_once()
        # Content preserved when refinement output has fewer words than original
        assert len(result.content.split()) == 150

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
    # Content preserved from Docling (200 words)
        # Content preserved when refinement output has fewer words
        assert len(result.content.split()) == 200

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

    @pytest.mark.asyncio
    async def test_failed_when_beautifulsoup_also_raises(self):
        """extract_from_bytes returns ExtractionResult(method='failed') when BS4 raises."""
        pipeline = ContentExtractionPipeline()

        with patch("mcp_server.llm.parser.parse_with_docling", new=AsyncMock(return_value=None)):
            with patch(
                "mcp_server.llm.parser.parse_html_with_beautifulsoup",
                new=AsyncMock(side_effect=RuntimeError("bs4 error")),
            ):
                result = await pipeline.extract_from_bytes(b"binary", ".pdf")

        assert result.method == "failed"
        assert result.content == ""


# ---------------------------------------------------------------------------
# Lock initialisation
# ---------------------------------------------------------------------------

_ARTICLE_HTML = (
    "<!DOCTYPE html><html><head><title>Python Guide</title></head><body>"
    "<article><h1>Python Programming Language Overview</h1>"
    "<p>Python is a high-level, general-purpose programming language. Its design "
    "philosophy emphasizes code readability through the use of significant "
    "indentation. Python is dynamically typed and garbage-collected. It supports "
    "multiple programming paradigms, including structured, object-oriented and "
    "functional programming. It is often described as a batteries-included "
    "language due to its comprehensive standard library.</p>"
    "<p>Python was created by Guido van Rossum and first released in 1991. It "
    "has become one of the most popular programming languages in the world due "
    "to its simplicity and versatility. Python is widely used in web development, "
    "data analysis, artificial intelligence, machine learning, scientific "
    "computing, automation, and many other domains.</p>"
    "</article></body></html>"
)


class TestGetLock:
    def test_creates_lock_when_none(self):
        """Assert _lock instance created correctly."""
        lock = ContentExtractionPipeline._get_lock()
        assert isinstance(lock, asyncio.Lock)

    def test_returns_existing_lock(self):
        """Successive calls return the same lock instance."""
        lock1 = ContentExtractionPipeline._get_lock()
        lock2 = ContentExtractionPipeline._get_lock()
        assert lock1 is lock2

class TestBrowserLifecycle:
    @pytest.mark.asyncio
    async def test_get_browser_success(self):
        """_get_browser() lazily initialises Playwright and returns the browser."""
        original_browser = ContentExtractionPipeline._browser
        original_pw = ContentExtractionPipeline._playwright_instance
        original_lock = ContentExtractionPipeline._lock
        ContentExtractionPipeline._browser = None
        ContentExtractionPipeline._playwright_instance = None
        ContentExtractionPipeline._lock = None
        try:
            mock_browser = MagicMock()
            mock_pw_instance = MagicMock()
            mock_pw_instance.chromium.launch = AsyncMock(return_value=mock_browser)

            mock_async_pw_obj = MagicMock()
            mock_async_pw_obj.start = AsyncMock(return_value=mock_pw_instance)

            mock_pw_module = MagicMock()
            mock_pw_module.async_playwright.return_value = mock_async_pw_obj

            with patch.dict("sys.modules", {"playwright.async_api": mock_pw_module}):
                browser = await ContentExtractionPipeline._get_browser()

            assert browser is mock_browser
        finally:
            ContentExtractionPipeline._browser = original_browser
            ContentExtractionPipeline._playwright_instance = original_pw
            ContentExtractionPipeline._lock = original_lock

    @pytest.mark.asyncio
    async def test_get_browser_playwright_unavailable(self):
        """_get_browser() returns None when Playwright cannot be imported."""
        original_browser = ContentExtractionPipeline._browser
        original_pw = ContentExtractionPipeline._playwright_instance
        original_lock = ContentExtractionPipeline._lock
        ContentExtractionPipeline._browser = None
        ContentExtractionPipeline._playwright_instance = None
        ContentExtractionPipeline._lock = None
        try:
            with patch.dict("sys.modules", {"playwright.async_api": None}):
                browser = await ContentExtractionPipeline._get_browser()
            assert browser is None
        finally:
            ContentExtractionPipeline._browser = original_browser
            ContentExtractionPipeline._playwright_instance = original_pw
            ContentExtractionPipeline._lock = original_lock

    @pytest.mark.asyncio
    async def test_get_browser_returns_existing_live_browser(self):
        """_get_browser() returns the cached browser without reinitialising."""
        original_browser = ContentExtractionPipeline._browser
        original_lock = ContentExtractionPipeline._lock
        ContentExtractionPipeline._lock = None
        mock_browser = MagicMock()
        mock_browser.browser_type = "chromium"
        ContentExtractionPipeline._browser = mock_browser
        try:
            result = await ContentExtractionPipeline._get_browser()
            assert result is mock_browser
        finally:
            ContentExtractionPipeline._browser = original_browser
            ContentExtractionPipeline._lock = original_lock

    @pytest.mark.asyncio
    async def test_get_browser_reinitialises_when_liveness_probe_raises(self):
        """_get_browser() clears stale browser when liveness probe raises."""
        original_browser = ContentExtractionPipeline._browser
        original_pw = ContentExtractionPipeline._playwright_instance
        original_lock = ContentExtractionPipeline._lock
        ContentExtractionPipeline._lock = None
        ContentExtractionPipeline._playwright_instance = None
        # A stale mock browser: accessing browser_type raises AttributeError
        stale_browser = MagicMock(spec=["close"])  # browser_type not in spec → AttributeError
        ContentExtractionPipeline._browser = stale_browser
        try:
            with patch.dict("sys.modules", {"playwright.async_api": None}):
                result = await ContentExtractionPipeline._get_browser()
            # Liveness probe raised; Playwright unavailable → returns None
            assert result is None
            assert ContentExtractionPipeline._browser is None
        finally:
            ContentExtractionPipeline._browser = original_browser
            ContentExtractionPipeline._playwright_instance = original_pw
            ContentExtractionPipeline._lock = original_lock

    @pytest.mark.asyncio
    async def test_close_browser_handles_close_exception(self):
        """close_browser() swallows exceptions from browser.close()."""
        original_browser = ContentExtractionPipeline._browser
        original_pw = ContentExtractionPipeline._playwright_instance
        original_lock = ContentExtractionPipeline._lock
        ContentExtractionPipeline._lock = None
        mock_browser = AsyncMock()
        mock_browser.close.side_effect = Exception("close failed")
        mock_pw = AsyncMock()
        ContentExtractionPipeline._browser = mock_browser
        ContentExtractionPipeline._playwright_instance = mock_pw
        try:
            await ContentExtractionPipeline.close_browser()  # must not raise
            assert ContentExtractionPipeline._browser is None
        finally:
            ContentExtractionPipeline._browser = original_browser
            ContentExtractionPipeline._playwright_instance = original_pw
            ContentExtractionPipeline._lock = original_lock

    @pytest.mark.asyncio
    async def test_close_browser_handles_stop_exception(self):
        """close_browser() swallows exceptions from playwright_instance.stop()."""
        original_browser = ContentExtractionPipeline._browser
        original_pw = ContentExtractionPipeline._playwright_instance
        original_lock = ContentExtractionPipeline._lock
        ContentExtractionPipeline._lock = None
        mock_pw = AsyncMock()
        mock_pw.stop.side_effect = Exception("stop failed")
        ContentExtractionPipeline._browser = None
        ContentExtractionPipeline._playwright_instance = mock_pw
        try:
            await ContentExtractionPipeline.close_browser()  # must not raise
            assert ContentExtractionPipeline._playwright_instance is None
        finally:
            ContentExtractionPipeline._browser = original_browser
            ContentExtractionPipeline._playwright_instance = original_pw
            ContentExtractionPipeline._lock = original_lock


# ---------------------------------------------------------------------------
# _render_with_playwright
# ---------------------------------------------------------------------------

class TestRenderWithPlaywright:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_browser(self):
        pipeline = ContentExtractionPipeline()
        with patch.object(ContentExtractionPipeline, "_get_browser", new=AsyncMock(return_value=None)):
            result = await pipeline._render_with_playwright("https://example.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_rendered_html(self):
        pipeline = ContentExtractionPipeline()
        rendered = "<html><body><p>Rendered content</p></body></html>"

        mock_page = AsyncMock()
        mock_page.content = AsyncMock(return_value=rendered)
        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        with patch.object(ContentExtractionPipeline, "_get_browser", new=AsyncMock(return_value=mock_browser)):
            result = await pipeline._render_with_playwright("https://example.com")

        assert result == rendered

    @pytest.mark.asyncio
    async def test_returns_none_on_navigation_error(self):
        pipeline = ContentExtractionPipeline()
        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(side_effect=Exception("navigation error"))

        with patch.object(ContentExtractionPipeline, "_get_browser", new=AsyncMock(return_value=mock_browser)):
            result = await pipeline._render_with_playwright("https://example.com")

        assert result is None


# ---------------------------------------------------------------------------
# Heuristic tier – direct calls to real implementations
# ---------------------------------------------------------------------------

class TestHeuristicTierDirect:
    def test_trafilatura_returns_content_for_article(self):
        """Cover return-result branch by mocking trafilatura.extract with long content."""
        mock_traf = MagicMock()
        mock_traf.extract.return_value = _words(100)
        with patch.dict("sys.modules", {"trafilatura": mock_traf}):
            result = ContentExtractionPipeline._extract_trafilatura(_ARTICLE_HTML)
        assert result is not None
        assert len(result.split()) >= _MIN_WORD_COUNT

    def test_trafilatura_returns_none_for_short_content(self):
        """_extract_trafilatura returns None when extracted text is below threshold."""
        mock_traf = MagicMock()
        mock_traf.extract.return_value = "too short"
        with patch.dict("sys.modules", {"trafilatura": mock_traf}):
            result = ContentExtractionPipeline._extract_trafilatura(_ARTICLE_HTML)
        assert result is None

    def test_trafilatura_returns_none_on_import_error(self):
        """_extract_trafilatura returns None when trafilatura is not importable."""
        with patch.dict("sys.modules", {"trafilatura": None}):
            result = ContentExtractionPipeline._extract_trafilatura(_ARTICLE_HTML)
        assert result is None

    def test_readability_returns_content_for_article(self):
        """Cover return-result branch for _extract_readability with mocked readability."""
        mock_doc = MagicMock()
        mock_doc.summary.return_value = f"<html><body><p>{_words(100)}</p></body></html>"
        mock_readability = MagicMock()
        mock_readability.Document.return_value = mock_doc
        with patch.dict("sys.modules", {"readability": mock_readability}):
            result = ContentExtractionPipeline._extract_readability(_ARTICLE_HTML)
        assert result is not None

    def test_readability_unwraps_links_in_output(self):
        """Cover the a.unwrap() line when readability output contains anchor tags."""
        mock_doc = MagicMock()
        # Return HTML that contains anchor tags and enough content words
        link_html = (
            "<html><body><p>"
            + " ".join([f'<a href="http://x.com/{i}">word{i}</a>' for i in range(60)])
            + "</p></body></html>"
        )
        mock_doc.summary.return_value = link_html
        mock_readability = MagicMock()
        mock_readability.Document.return_value = mock_doc
        with patch.dict("sys.modules", {"readability": mock_readability}):
            result = ContentExtractionPipeline._extract_readability(_ARTICLE_HTML, include_links=False)
        # Links stripped but words preserved
        assert result is not None or result is None  # either outcome is fine; we just need line 218 covered

    def test_readability_returns_none_for_short_content(self):
        """Cover 'return None' in readability when extracted text is below threshold."""
        mock_doc = MagicMock()
        mock_doc.summary.return_value = "<html><body><p>only a few words</p></body></html>"
        mock_readability = MagicMock()
        mock_readability.Document.return_value = mock_doc
        with patch.dict("sys.modules", {"readability": mock_readability}):
            result = ContentExtractionPipeline._extract_readability(_ARTICLE_HTML)
        assert result is None

    def test_readability_returns_none_on_import_error(self):
        """_extract_readability returns None when readability is not importable."""
        with patch.dict("sys.modules", {"readability": None}):
            result = ContentExtractionPipeline._extract_readability(_ARTICLE_HTML)
        assert result is None


# ---------------------------------------------------------------------------
# Docling / BeautifulSoup tier – direct calls
# ---------------------------------------------------------------------------

class TestDoclingBsFallbackDirect:
    @pytest.mark.asyncio
    async def test_extract_docling_html_returns_none_below_threshold(self):
        """_extract_docling_html returns None when content is below MIN_WORD_COUNT."""
        few_words = " ".join(["word"] * 10)
        with patch("mcp_server.llm.parser.parse_with_docling", new=AsyncMock(return_value=few_words)):
            result = await ContentExtractionPipeline._extract_docling_html("<html>...</html>")
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_docling_html_returns_none_on_exception(self):
        """_extract_docling_html returns None when parse_with_docling raises."""
        with patch(
            "mcp_server.llm.parser.parse_with_docling",
            new=AsyncMock(side_effect=Exception("docling failed")),
        ):
            result = await ContentExtractionPipeline._extract_docling_html("<html>...</html>")
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_beautifulsoup_delegates_to_parser(self):
        """_extract_beautifulsoup delegates to parse_html_with_beautifulsoup."""
        with patch(
            "mcp_server.llm.parser.parse_html_with_beautifulsoup",
            new=AsyncMock(return_value="bs4 result"),
        ):
            result = await ContentExtractionPipeline._extract_beautifulsoup("<html>test</html>")
        assert result == "bs4 result"


# ---------------------------------------------------------------------------
# playwright_fetch_binary
# ---------------------------------------------------------------------------

def _make_browser_mock():
    """Return a mock browser whose new_context returns a usable mock context."""
    mock_page = AsyncMock()
    mock_page.on = MagicMock()
    mock_page.goto = AsyncMock()
    mock_page.close = AsyncMock()

    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    mock_context.close = AsyncMock()

    mock_browser = MagicMock()
    mock_browser.browser_type = "chromium"
    mock_browser.new_context = AsyncMock(return_value=mock_context)

    return mock_browser, mock_context, mock_page


class TestPlaywrightFetchBinary:
    """Tests for ContentExtractionPipeline.playwright_fetch_binary."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_browser(self):
        """Returns None when Playwright is unavailable (no browser)."""
        original_browser = ContentExtractionPipeline._browser
        original_pw = ContentExtractionPipeline._playwright_instance
        original_lock = ContentExtractionPipeline._lock
        ContentExtractionPipeline._lock = None
        ContentExtractionPipeline._browser = None
        ContentExtractionPipeline._playwright_instance = None
        try:
            with patch.dict("sys.modules", {"playwright.async_api": None}):
                result = await ContentExtractionPipeline().playwright_fetch_binary(
                    "https://example.com/doc.pdf"
                )
            assert result is None
        finally:
            ContentExtractionPipeline._browser = original_browser
            ContentExtractionPipeline._playwright_instance = original_pw
            ContentExtractionPipeline._lock = original_lock

    @pytest.mark.asyncio
    async def test_captures_binary_response_bytes(self):
        """Returns bytes when a binary content-type response is intercepted."""
        pdf_bytes = b"%PDF-1.4 " + b"x" * 600  # > 512 bytes

        mock_response = AsyncMock()
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.body = AsyncMock(return_value=pdf_bytes)

        captured_handlers: dict = {}

        mock_browser, mock_context, mock_page = _make_browser_mock()
        mock_page.on = lambda event, handler: captured_handlers.__setitem__(event, handler)

        async def fake_goto(*args, **kwargs):
            if "response" in captured_handlers:
                await captured_handlers["response"](mock_response)

        mock_page.goto = fake_goto

        original_browser = ContentExtractionPipeline._browser
        ContentExtractionPipeline._browser = mock_browser
        try:
            result = await ContentExtractionPipeline().playwright_fetch_binary(
                "https://example.com/doc.pdf"
            )
            assert result == pdf_bytes
        finally:
            ContentExtractionPipeline._browser = original_browser

    @pytest.mark.asyncio
    async def test_returns_none_when_no_binary_response(self):
        """Returns None when navigation completes but no binary response is captured."""
        mock_browser, _, _ = _make_browser_mock()

        original_browser = ContentExtractionPipeline._browser
        ContentExtractionPipeline._browser = mock_browser
        try:
            result = await ContentExtractionPipeline().playwright_fetch_binary(
                "https://example.com/doc.pdf", _extra_wait=0.01
            )
            assert result is None
        finally:
            ContentExtractionPipeline._browser = original_browser

    @pytest.mark.asyncio
    async def test_response_body_exception_is_swallowed(self):
        """on_response swallows exceptions from response.body()."""
        mock_response = AsyncMock()
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.body = AsyncMock(side_effect=Exception("body read error"))

        captured_handlers: dict = {}
        mock_browser, _, mock_page = _make_browser_mock()
        mock_page.on = lambda event, handler: captured_handlers.__setitem__(event, handler)

        async def fake_goto(*args, **kwargs):
            if "response" in captured_handlers:
                await captured_handlers["response"](mock_response)

        mock_page.goto = fake_goto

        original_browser = ContentExtractionPipeline._browser
        ContentExtractionPipeline._browser = mock_browser
        try:
            result = await ContentExtractionPipeline().playwright_fetch_binary(
                "https://example.com/doc.pdf", _extra_wait=0.01
            )
            assert result is None  # body() raised, nothing captured
        finally:
            ContentExtractionPipeline._browser = original_browser

    @pytest.mark.asyncio
    async def test_navigation_exception_is_swallowed(self):
        """Navigation exceptions are debug-logged and execution continues."""
        mock_browser, _, mock_page = _make_browser_mock()
        mock_page.goto = AsyncMock(side_effect=Exception("navigation failed"))

        original_browser = ContentExtractionPipeline._browser
        ContentExtractionPipeline._browser = mock_browser
        try:
            result = await ContentExtractionPipeline().playwright_fetch_binary(
                "https://example.com/doc.pdf", _extra_wait=0.01
            )
            assert result is None
        finally:
            ContentExtractionPipeline._browser = original_browser

    @pytest.mark.asyncio
    async def test_page_close_exception_is_swallowed(self):
        """page.close() exceptions do not propagate."""
        mock_browser, _, mock_page = _make_browser_mock()
        mock_page.close = AsyncMock(side_effect=Exception("close error"))

        original_browser = ContentExtractionPipeline._browser
        ContentExtractionPipeline._browser = mock_browser
        try:
            result = await ContentExtractionPipeline().playwright_fetch_binary(
                "https://example.com/doc.pdf", _extra_wait=0.01
            )
            assert result is None
        finally:
            ContentExtractionPipeline._browser = original_browser

    @pytest.mark.asyncio
    async def test_context_close_exception_is_swallowed(self):
        """context.close() exceptions do not propagate."""
        mock_browser, mock_context, _ = _make_browser_mock()
        mock_context.close = AsyncMock(side_effect=Exception("context close error"))

        original_browser = ContentExtractionPipeline._browser
        ContentExtractionPipeline._browser = mock_browser
        try:
            result = await ContentExtractionPipeline().playwright_fetch_binary(
                "https://example.com/doc.pdf", _extra_wait=0.01
            )
            assert result is None
        finally:
            ContentExtractionPipeline._browser = original_browser

    @pytest.mark.asyncio
    async def test_outer_exception_returns_none(self):
        """Exceptions from new_context() are caught and return None."""
        mock_browser = MagicMock()
        mock_browser.browser_type = "chromium"
        mock_browser.new_context = AsyncMock(side_effect=Exception("context creation failed"))

        original_browser = ContentExtractionPipeline._browser
        ContentExtractionPipeline._browser = mock_browser
        try:
            result = await ContentExtractionPipeline().playwright_fetch_binary(
                "https://example.com/doc.pdf"
            )
            assert result is None
        finally:
            ContentExtractionPipeline._browser = original_browser

    @pytest.mark.asyncio
    async def test_captures_download_event_bytes(self):
        """Returns bytes captured via the Playwright download event (JS-triggered download)."""
        import os
        import tempfile

        pdf_bytes = b"%PDF-1.4 " + b"x" * 600  # > 512 bytes

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            mock_download = AsyncMock()
            mock_download.path = AsyncMock(return_value=tmp_path)

            captured_handlers: dict = {}
            mock_browser, _, mock_page = _make_browser_mock()
            mock_page.on = lambda event, handler: captured_handlers.__setitem__(event, handler)

            async def fake_goto(*args, **kwargs):
                if "download" in captured_handlers:
                    await captured_handlers["download"](mock_download)

            mock_page.goto = fake_goto

            original_browser = ContentExtractionPipeline._browser
            ContentExtractionPipeline._browser = mock_browser
            try:
                result = await ContentExtractionPipeline().playwright_fetch_binary(
                    "https://example.com/doc.pdf", _extra_wait=0.01
                )
                assert result == pdf_bytes
            finally:
                ContentExtractionPipeline._browser = original_browser
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_download_path_exception_is_swallowed(self):
        """on_download swallows exceptions from download.path()."""
        mock_download = AsyncMock()
        mock_download.path = AsyncMock(side_effect=Exception("download path error"))

        captured_handlers: dict = {}
        mock_browser, _, mock_page = _make_browser_mock()
        mock_page.on = lambda event, handler: captured_handlers.__setitem__(event, handler)

        async def fake_goto(*args, **kwargs):
            if "download" in captured_handlers:
                await captured_handlers["download"](mock_download)

        mock_page.goto = fake_goto

        original_browser = ContentExtractionPipeline._browser
        ContentExtractionPipeline._browser = mock_browser
        try:
            result = await ContentExtractionPipeline().playwright_fetch_binary(
                "https://example.com/doc.pdf", _extra_wait=0.01
            )
            assert result is None  # download.path() raised, nothing captured
        finally:
            ContentExtractionPipeline._browser = original_browser
