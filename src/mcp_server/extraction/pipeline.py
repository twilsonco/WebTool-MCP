"""
Multi-tiered content extraction pipeline.

Implements a cascading extraction strategy with graceful tier fallback:

  Tier 0  – Firecrawl (optional, when USE_FIRECRAWL=true): primary scraping
             via the Firecrawl API; if it fails or returns thin content (<200 words),
             the entire legacy pipeline runs as one fallback block.
  Tier 1  – Dynamic Rendering (Playwright): executes JavaScript and waits for
             the page to fully hydrate before handing the DOM to Tier 2.
  Tier 2  – Heuristic/Text-Density (Trafilatura → Readability-lxml): fast
             boilerplate removal via text-density analysis and DOM article
             extraction; preferred for traditional HTML pages.
  Tier 3  – Layout-Aware (Docling): handles complex document structures,
             tables, and multi-column layouts that defeat heuristic parsers.
  Tier 4  – Fallback (BeautifulSoup): always-succeeds minimal converter used
             only when all preceding tiers fail or return insufficient content.
  Opt.    – Cognitive Refinement (LLM): optional final pass that uses the
             configured LLM to strip remaining formatting noise and return
             clean, semantic Markdown when content quality is still poor.

Resources (Playwright browser, Docling converter) are lazily initialised as
class-level singletons and reused across calls to minimise per-request
overhead.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

USE_FIRECRAWL = os.getenv("USE_FIRECRAWL", "false").lower() == "true"

# Minimum word count considered a successful extraction result.
_MIN_WORD_COUNT = 50

# Word count above which we do not escalate to the next tier.
_RICH_WORD_COUNT = 200

# LLM cognitive-refinement threshold: only refine when content is below this.
_LLM_TRIGGER_WORD_COUNT = 100


@dataclass
class ExtractionResult:
    """Result produced by a pipeline extraction attempt."""

    content: str
    method: str  # e.g. "playwright+trafilatura", "static+beautifulsoup"

    @property
    def word_count(self) -> int:
        return len(self.content.split()) if self.content else 0


class ContentExtractionPipeline:
    """
    Multi-tiered web content extraction pipeline (class-level singleton state).

    Resources are shared across instances so that a persistent Playwright
    browser process is reused rather than spawned per-request.
    """

    # Class-level shared state -------------------------------------------------
    _lock: Optional[asyncio.Lock] = None
    _playwright_instance = None
    _browser = None

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                        #
    # ---------------------------------------------------------------------- #

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        """Return (lazily initialised) async lock guarding shared browser state."""
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    # ---------------------------------------------------------------------- #
    # Tier 1 – Dynamic Rendering (Playwright)                                 #
    # ---------------------------------------------------------------------- #

    @classmethod
    async def _get_browser(cls):
        """Lazily initialise and return the shared headless Chromium browser.

        Returns None when Playwright is unavailable or the browser cannot be
        launched (e.g. Chromium not installed).  The caller must handle None.
        """
        async with cls._get_lock():
            if cls._browser is not None:
                try:
                    # Lightweight liveness probe – just access an attribute.
                    _ = cls._browser.browser_type
                    return cls._browser
                except Exception:
                    cls._browser = None
                    cls._playwright_instance = None

            try:
                from playwright.async_api import async_playwright

                cls._playwright_instance = await async_playwright().start()
                cls._browser = await cls._playwright_instance.chromium.launch(
                    headless=True
                )
                logger.info("Playwright Chromium browser initialised")
            except Exception as exc:
                logger.warning("Playwright unavailable: %s", exc)
                cls._browser = None
                cls._playwright_instance = None

            return cls._browser

    @classmethod
    async def close_browser(cls) -> None:
        """Gracefully shut down the shared Playwright browser.

        Call this on server shutdown to release the Chromium process.
        """
        async with cls._get_lock():
            if cls._browser is not None:
                try:
                    await cls._browser.close()
                except Exception:
                    pass
                cls._browser = None
            if cls._playwright_instance is not None:
                try:
                    await cls._playwright_instance.stop()
                except Exception:
                    pass
                cls._playwright_instance = None

    # ---------------------------------------------------------------------- #
    # Tier 0 – Firecrawl (when USE_FIRECRAWL=true)                           #
    # ---------------------------------------------------------------------- #

    @classmethod
    async def _extract_with_firecrawl(cls, url: str) -> Optional[ExtractionResult]:
        """Try Firecrawl scrape as primary extraction method.

        Returns an ExtractionResult if Firecrawl succeeds with at least
        _RICH_WORD_COUNT words; otherwise returns None to trigger the legacy pipeline.
        """
        if not USE_FIRECRAWL:
            return None
        try:
            from mcp_server.extraction import get_firecrawl_client

            client = get_firecrawl_client()
            if client is None:
                return None
            result = await client.scrape(url)
            if result and result.word_count >= _RICH_WORD_COUNT:
                return result
            return None
        except Exception as exc:
            logger.debug("Firecrawl extraction failed: %s", exc)
            return None

    async def _render_with_playwright(
        self, url: str, timeout: float = 20.0
    ) -> Optional[str]:
        """Render *url* with Playwright and return the fully-hydrated DOM HTML.

        Waits for ``networkidle`` so that JavaScript-rendered content finishes
        loading before the DOM is captured.  Returns ``None`` on any failure
        (browser unavailable, navigation error, or timeout).
        """
        try:
            browser = await self._get_browser()
            if browser is None:
                return None

            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )
            page = await context.new_page()
            try:
                await page.goto(
                    url, wait_until="networkidle", timeout=int(timeout * 1000)
                )
                return await page.content()
            finally:
                await page.close()
                await context.close()
        except Exception as exc:
            logger.warning("Playwright rendering failed for %s: %s", url, exc)
            return None

    # ---------------------------------------------------------------------- #
    # Tier 2 – Heuristic Extraction (Trafilatura / Readability-lxml)          #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _extract_trafilatura(
        html: str, include_links: bool = True
    ) -> Optional[str]:
        """Extract main content using Trafilatura's text-density algorithm.

        Returns ``None`` when Trafilatura is unavailable, extraction fails, or
        the result falls below the minimum word threshold.
        """
        try:
            import trafilatura

            result = trafilatura.extract(
                html,
                include_links=include_links,
                include_tables=True,
                no_fallback=False,
                favor_recall=True,
                output_format="markdown",
            )
            if result and len(result.split()) >= _MIN_WORD_COUNT:
                return result
            return None
        except Exception as exc:
            logger.debug("Trafilatura extraction failed: %s", exc)
            return None

    @staticmethod
    def _extract_readability(
        html: str, include_links: bool = True
    ) -> Optional[str]:
        """Extract main content using readability-lxml's article extraction.

        Returns ``None`` when the library is unavailable, extraction fails, or
        the result falls below the minimum word threshold.
        """
        try:
            from readability import Document
            from markdownify import markdownify as md
            from bs4 import BeautifulSoup

            doc = Document(html)
            content_html = doc.summary()

            if not include_links:
                soup = BeautifulSoup(content_html, "html.parser")
                for a in soup.find_all("a"):
                    a.unwrap()
                content_html = str(soup)

            result = md(content_html)
            if result and len(result.split()) >= _MIN_WORD_COUNT:
                return result
            return None
        except Exception as exc:
            logger.debug("Readability extraction failed: %s", exc)
            return None

    # ---------------------------------------------------------------------- #
    # Tier 3 – Layout-Aware (Docling)                                         #
    # ---------------------------------------------------------------------- #

    @staticmethod
    async def _extract_docling_html(
        html: str, include_links: bool = True
    ) -> Optional[str]:
        """Attempt Docling extraction on raw HTML content.

        Returns ``None`` when Docling is unavailable, fails, or the result is
        below the minimum word threshold.
        """
        try:
            from mcp_server.llm.parser import parse_with_docling

            result = await parse_with_docling(
                html.encode("utf-8"), ".html", include_links
            )
            if result and len(result.split()) >= _MIN_WORD_COUNT:
                return result
            return None
        except Exception as exc:
            logger.debug("Docling HTML extraction failed: %s", exc)
            return None

    # ---------------------------------------------------------------------- #
    # Tier 4 – Fallback (BeautifulSoup)                                       #
    # ---------------------------------------------------------------------- #

    @staticmethod
    async def _extract_beautifulsoup(
        html: str, include_links: bool = True
    ) -> str:
        """Extract content using BeautifulSoup (always succeeds).

        Performs minimal structural filtering via markdownify; used only when
        all preceding tiers fail or return insufficient content.
        """
        from mcp_server.llm.parser import parse_html_with_beautifulsoup

        return await parse_html_with_beautifulsoup(html, include_links)

    # ---------------------------------------------------------------------- #
    # Optional – Cognitive Refinement (LLM)                                   #
    # ---------------------------------------------------------------------- #

    @staticmethod
    async def _refine_with_llm(content: str, llm_manager) -> Optional[str]:
        """Apply LLM-based semantic refinement to noisy extracted content.

        Passes the content (capped at 4 000 characters) to the configured LLM
        and asks it to strip boilerplate and return clean Markdown.  Returns
        ``None`` on any LLM failure so the caller can keep the original text.

        Args:
            content: The extracted text to refine.
            llm_manager: An ``LLMManager`` instance (or any object with a
                compatible ``complete(prompt, system_prompt)`` coroutine).
        """
        try:
            refined = await llm_manager.complete(
                (
                    "Extract and clean the main content from the text below. "
                    "Remove navigation menus, sidebars, footers, cookie banners, "
                    "ads, and other boilerplate. Return only the core body text "
                    "as clean, well-structured Markdown:\n\n"
                    f"{content[:4000]}"
                ),
                system_prompt=(
                    "You are a content extraction assistant. "
                    "Output only the main article/body content as clean Markdown. "
                    "Do not add commentary, introductions, or meta-notes."
                ),
            )
            return refined.strip() if refined else None
        except Exception as exc:
            logger.warning("LLM refinement failed: %s", exc)
            return None

    # ---------------------------------------------------------------------- #
    # Main entry points                                                       #
    # ---------------------------------------------------------------------- #

    async def extract_from_html(
        self,
        html: str,
        url: str,
        include_links: bool = True,
        use_playwright: bool = True,
        use_llm_refinement: Optional[bool] = None,
        llm_manager=None,
    ) -> ExtractionResult:
        """Extract content from an HTML page through the full tier pipeline.

        The pipeline attempts each tier in sequence and stops as soon as it
        obtains a result above ``_RICH_WORD_COUNT``.  If a tier fails or
        returns insufficient content the next tier is tried automatically.

        When USE_FIRECRAWL=true, Firecrawl scrape is attempted first (Tier 0).
        If Firecrawl succeeds with rich content (>=200 words), it returns immediately.
        On failure or thin content, the entire legacy pipeline runs as one fallback block.

        Args:
            html: Pre-fetched static HTML used when Playwright is disabled or
                  fails; always available as the Tier-1 fallback source.
            url: Original page URL passed to Playwright for dynamic rendering.
            include_links: Preserve anchor ``href`` attributes in the output.
            use_playwright: Attempt Tier-1 dynamic rendering (default ``True``).
            use_llm_refinement: Apply optional LLM cognitive refinement when
                content quality remains below ``_LLM_TRIGGER_WORD_COUNT``.
                When None, defaults to True for textual content.
            llm_manager: ``LLMManager`` instance required when
                ``use_llm_refinement`` is ``True``; ignored otherwise.

        Returns:
            :class:`ExtractionResult` carrying the best extracted content and
            a description of the tier(s) used (e.g. ``"playwright+trafilatura"``).
        """
        # --- Tier 0: Firecrawl (when USE_FIRECRAWL=true) ------------------
        if USE_FIRECRAWL:
            firecrawl_result = await self._extract_with_firecrawl(url)
            if firecrawl_result is not None:
                return firecrawl_result

        # --- Legacy pipeline fallback (runs as one block when Firecrawl fails/thins out) ---
        working_html = html
        method_prefix = "static"

        if use_playwright:
            rendered = await self._render_with_playwright(url)
            if rendered:
                working_html = rendered
                method_prefix = "playwright"

        # --- Tier 2: Heuristic Extraction (Trafilatura → Readability) -----
        content: Optional[str] = self._extract_trafilatura(working_html, include_links)
        method = f"{method_prefix}+trafilatura"

        if content is None or len(content.split()) < _MIN_WORD_COUNT:
            readability_content = self._extract_readability(working_html, include_links)
            if readability_content and (
                content is None
                or len(readability_content.split()) > len(content.split())
            ):
                content = readability_content
                method = f"{method_prefix}+readability"

        # Early exit when content is already rich enough.
        if content is not None and len(content.split()) >= _RICH_WORD_COUNT:
            return ExtractionResult(content=content, method=method)

        # --- Tier 3: Layout-Aware (Docling) --------------------------------
        docling_content = await self._extract_docling_html(working_html, include_links)
        if docling_content and (
            content is None or len(docling_content.split()) > len(content.split())
        ):
            content = docling_content
            method = f"{method_prefix}+docling"

        if content is not None and len(content.split()) >= _RICH_WORD_COUNT:
            return ExtractionResult(content=content, method=method)

        # --- Tier 4: BeautifulSoup (always succeeds) ----------------------
        bs_content = await self._extract_beautifulsoup(working_html, include_links)
        if bs_content and (
            content is None or len(bs_content.split()) > len(content.split())
        ):
            content = bs_content
            method = f"{method_prefix}+beautifulsoup"

        content = content or ""

        # --- Optional: LLM Cognitive Refinement ---------------------------
        if (
            use_llm_refinement
            and llm_manager is not None
            and content
            and len(content.split()) < _LLM_TRIGGER_WORD_COUNT
        ):
            refined = await self._refine_with_llm(content, llm_manager)
            if refined and len(refined.split()) > len(content.split()):
                content = refined
                method = f"{method}+llm"

        return ExtractionResult(content=content, method=method)

    async def extract_from_bytes(
        self,
        content_bytes: bytes,
        file_extension: str,
        include_links: bool = True,
    ) -> ExtractionResult:
        """Extract content from a binary document (PDF, DOCX, images, …).

        Uses Docling as the primary parser.  Falls back to BeautifulSoup
        (treating the bytes as UTF-8 HTML) on Docling failure.

        Args:
            content_bytes: Raw binary content of the document.
            file_extension: Lower-case file extension including the dot
                (e.g. ``".pdf"``, ``".docx"``).
            include_links: Preserve anchor ``href`` attributes in the output.

        Returns:
            :class:`ExtractionResult` with ``method`` set to ``"docling"`` on
            success or ``"beautifulsoup"`` / ``"failed"`` on fallback.
        """
        from mcp_server.llm.parser import parse_with_docling, parse_html_with_beautifulsoup

        result = await parse_with_docling(content_bytes, file_extension, include_links)
        if result and len(result.split()) >= _MIN_WORD_COUNT:
            return ExtractionResult(content=result, method="docling")

        # Fallback: attempt to decode bytes as UTF-8 HTML/text.
        try:
            html = content_bytes.decode("utf-8", errors="ignore")
            bs_result = await parse_html_with_beautifulsoup(html, include_links)
            return ExtractionResult(content=bs_result or "", method="beautifulsoup")
        except Exception:
            return ExtractionResult(content="", method="failed")

    async def capture_screenshot(
        self,
        url: str,
        full_page: bool = False,
        quality: Optional[int] = None,
        viewport_width: Optional[int] = None,
        viewport_height: Optional[int] = None,
    ) -> Optional[str]:
        """
        Navigate to URL and capture a screenshot as base64 PNG.

        When USE_FIRECRAWL=true and any of the extended options (full_page, quality,
        viewport_width, viewport_height) are set, delegates to Firecrawl's screenshot
        API instead of Playwright.

        Args:
            url: The URL to navigate to.
            full_page: Capture the entire scrollable page (default False).
            quality: Image quality 1-100 for JPEG screenshots (default None).
            viewport_width: Browser viewport width in pixels (default None).
            viewport_height: Browser viewport height in pixels (default None).

        Returns:
            Base64-encoded PNG image, or None on failure.
        """
        if USE_FIRECRAWL and any(v is not None for v in [full_page, quality, viewport_width, viewport_height]):
            return await self._capture_screenshot_firecrawl(
                url, full_page=full_page, quality=quality,
                viewport_width=viewport_width, viewport_height=viewport_height
            )

        try:
            browser = await self._get_browser()
            if browser is None:
                return None

            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=20000)
                screenshot_bytes = await page.screenshot(format="png", encoding="base64")
                return screenshot_bytes
            finally:
                await page.close()
                await context.close()
        except Exception as exc:
            logger.warning("Screenshot capture failed for %s: %s", url, exc)
            return None

    async def _capture_screenshot_firecrawl(
        self,
        url: str,
        full_page: bool = False,
        quality: Optional[int] = None,
        viewport_width: Optional[int] = None,
        viewport_height: Optional[int] = None,
    ) -> Optional[str]:
        """Capture screenshot using Firecrawl API.

        Returns base64-encoded image string or None on failure.
        """
        try:
            from mcp_server.extraction import get_firecrawl_client

            client = get_firecrawl_client()
            if client is None:
                return None

            screenshot_data = await client.screenshot(
                url,
                full_page=full_page,
                quality=quality,
                width=viewport_width,
                height=viewport_height,
            )
            return screenshot_data
        except Exception as exc:
            logger.debug("Firecrawl screenshot failed: %s", exc)
            return None

    async def playwright_fetch_binary(
        self, url: str, timeout: float = 30.0, _extra_wait: Optional[float] = None
    ) -> Optional[bytes]:
        """Navigate to *url* with Playwright and capture any binary document payload.

        Designed for URLs that serve an HTML loading page before redirecting to a
        binary document (e.g. a PDF download after a JavaScript delay).  Intercepts
        both HTTP responses and browser download events, returning the bytes of the
        largest captured payload.

        Returns ``None`` when Playwright is unavailable or no binary payload is
        captured within *timeout* seconds.
        """
        _BINARY_PREFIXES = (
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats",
            "application/vnd.ms-",
            "application/octet-stream",
        )
        # After networkidle, wait up to this long for JS-triggered downloads.
        extra_wait = _extra_wait if _extra_wait is not None else min(timeout / 3, 10.0)
        try:
            browser = await self._get_browser()
            if browser is None:
                return None

            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                accept_downloads=True,
            )
            captured: list[bytes] = []
            download_complete = asyncio.Event()
            downloaded_data: list[bytes] = []

            async def on_response(response) -> None:
                ct = response.headers.get("content-type", "")
                if any(ct.startswith(prefix) for prefix in _BINARY_PREFIXES):
                    try:
                        body = await response.body()
                        if body and len(body) > 512:
                            captured.append(body)
                    except Exception:
                        pass

            async def on_download(download) -> None:
                try:
                    path = await download.path()
                    if path:
                        def _read_file() -> bytes:
                            with open(path, "rb") as f:
                                return f.read()
                        data = await asyncio.to_thread(_read_file)
                        if data and len(data) > 512:
                            downloaded_data.append(data)
                            download_complete.set()
                except Exception:
                    pass

            page = await context.new_page()
            page.on("response", on_response)
            page.on("download", on_download)

            try:
                try:
                    await page.goto(
                        url, wait_until="networkidle", timeout=int(timeout * 1000)
                    )
                except Exception as exc:
                    logger.debug(
                        "Playwright navigation ended early (expected for binary downloads): %s",
                        exc,
                    )
                # Wait for any JS-triggered downloads that may start after page load.
                if not captured and not downloaded_data:
                    try:
                        await asyncio.wait_for(download_complete.wait(), timeout=extra_wait)
                    except asyncio.TimeoutError:
                        pass
                # Yield once so any pending response callbacks can complete.
                await asyncio.sleep(0)
            finally:
                try:
                    await page.close()
                except Exception:
                    pass
                try:
                    await context.close()
                except Exception:
                    pass

            all_candidates = captured + downloaded_data
            return max(all_candidates, key=len) if all_candidates else None
        except Exception as exc:
            logger.warning("Playwright binary fetch failed for %s: %s", url, exc)
            return None
