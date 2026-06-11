"""
Async client for Firecrawl self-hosted API.

Firecrawl is a web scraping service that handles JavaScript rendering,
rate limiting, and extraction. This client wraps the Firecrawl API
for use in the content extraction pipeline.
"""

import os
from typing import Optional, Any
import logging

import httpx

from .pipeline import ExtractionResult

logger = logging.getLogger(__name__)

FIRECRAWL_API_URL_ENV = "FIRECRAWL_API_URL"
FIRECRAWL_API_KEY_ENV = "FIRECRAWL_API_KEY"

DEFAULT_FIRECRAWL_URL = "http://localhost:3002"


class FirecrawlClient:
    """Async client for Firecrawl self-hosted API."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        """
        Initialize the Firecrawl client.

        Args:
            api_url: Base URL for the Firecrawl API. Defaults to
                FIRECRAWL_API_URL env var or http://localhost:3002.
            api_key: Optional API key for authentication.
                Defaults to FIRECRAWL_API_KEY env var.
            timeout: Request timeout in seconds (default 60).
        """
        self.api_url = (api_url or os.environ.get(FIRECRAWL_API_URL_ENV) or DEFAULT_FIRECRAWL_URL).rstrip("/")
        self.api_key = api_key or os.environ.get(FIRECRAWL_API_KEY_ENV)
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _build_headers(self) -> dict[str, str]:
        """Build request headers including auth if configured."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def scrape(
        self,
        url: str,
        formats: list[str] | None = None,
        only_main_content: bool = True,
        actions: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Optional[ExtractionResult]:
        """
        Scrape a single URL and extract content.

        Args:
            url: The URL to scrape.
            formats: Output formats (default ["markdown"]).
            only_main_content: Extract only main content (default True).
            actions: Pre-scrape browser actions to perform.
            **kwargs: Additional ScrapeOptions fields.

        Returns:
            ExtractionResult on success, None on failure.
        """
        if formats is None:
            formats = ["markdown"]

        payload: dict[str, Any] = {
            "url": url,
            "formats": formats,
            "onlyMainContent": only_main_content,
        }

        if actions:
            payload["actions"] = actions

        payload.update(kwargs)

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.api_url}/v1/scrape",
                json=payload,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            data = response.json()

            content = self._extract_content_from_response(data, formats)
            if content:
                return ExtractionResult(content=content, method="firecrawl")
            return None

        except httpx.HTTPStatusError as e:
            logger.warning("Firecrawl scrape HTTP error for %s: %s", url, e.response.status_code)
            return None
        except Exception as e:
            logger.warning("Firecrawl scrape failed for %s: %s", url, str(e))
            return None

    async def screenshot(
        self,
        url: str,
        full_page: bool = False,
        format: str = "png",
        quality: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Optional[str]:
        """
        Capture a screenshot of the given URL.

        Args:
            url: The URL to capture.
            full_page: If True, capture the entire scrollable page.
            format: Image format ("png" or "jpeg", default "png").
            quality: Image quality 1-100 for JPEG screenshots.
            width: Viewport width in pixels.
            height: Viewport height in pixels.

        Returns:
            Base64-encoded image string on success, None on failure.
        """
        payload: dict[str, Any] = {
            "url": url,
            "fullPage": full_page,
            "format": format,
        }

        if quality is not None:
            payload["quality"] = quality
        if width is not None:
            payload["width"] = width
        if height is not None:
            payload["height"] = height

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.api_url}/v1/screenshot",
                json=payload,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and "screenshot" in data:
                return data["screenshot"]
            return None

        except httpx.HTTPStatusError as e:
            logger.warning("Firecrawl screenshot HTTP error for %s: %s", url, e.response.status_code)
            return None
        except Exception as e:
            logger.warning("Firecrawl screenshot failed for %s: %s", url, str(e))
            return None

    def _extract_content_from_response(
        self, data: dict[str, Any], formats: list[str]
    ) -> Optional[str]:
        """Extract content from Firecrawl response based on requested formats."""
        if not isinstance(data, dict):
            return None

        for fmt in formats:
            key = fmt
            if fmt == "markdown":
                key = "markdown"
            elif fmt == "html":
                key = "html"

            if key in data and data[key]:
                content = data[key]
                if isinstance(content, str) and len(content.split()) >= 10:
                    return content

        if "content" in data and data["content"]:
            return data["content"]

        return None

    async def batch_scrape(
        self,
        urls: list[str],
        formats: list[str] | None = None,
        only_main_content: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Submit a batch scrape job.

        Args:
            urls: List of URLs to scrape.
            formats: Output formats (default ["markdown"]).
            only_main_content: Extract only main content (default True).
            **kwargs: Additional batch options.

        Returns:
            Dict with job_id for status polling, or empty dict on failure.
        """
        if formats is None:
            formats = ["markdown"]

        payload: dict[str, Any] = {
            "urls": urls,
            "formats": formats,
            "onlyMainContent": only_main_content,
        }
        payload.update(kwargs)

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.api_url}/v1/batch/scrape",
                json=payload,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.warning("Firecrawl batch scrape HTTP error: %s", e.response.status_code)
            return {}
        except Exception as e:
            logger.warning("Firecrawl batch scrape failed: %s", str(e))
            return {}

    async def get_batch_status(self, job_id: str) -> dict[str, Any]:
        """
        Get the status of a batch scrape job.

        Args:
            job_id: The job ID returned from batch_scrape.

        Returns:
            Dict with job status and results if complete.
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.api_url}/v1/batch/{job_id}",
                headers=self._build_headers(),
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.warning("Firecrawl batch status HTTP error for %s: %s", job_id, e.response.status_code)
            return {}
        except Exception as e:
            logger.warning("Firecrawl batch status failed for %s: %s", job_id, str(e))
            return {}

    async def map_site(self, url: str, search_depth: int = 1) -> list[str]:
        """
        Discover URLs on a website.

        Args:
            url: The root URL to start crawling from.
            search_depth: How deep to crawl (default 1).

        Returns:
            List of discovered URLs.
        """
        payload = {
            "url": url,
            "searchDepth": search_depth,
        }

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.api_url}/v1/map",
                json=payload,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and "urls" in data:
                return data["urls"]
            if isinstance(data, list):
                return data
            return []

        except httpx.HTTPStatusError as e:
            logger.warning("Firecrawl map_site HTTP error for %s: %s", url, e.response.status_code)
            return []
        except Exception as e:
            logger.warning("Firecrawl map_site failed for %s: %s", url, str(e))
            return []

    async def crawl_async(
        self,
        url: str,
        search_depth: int = 1,
        max_depth: int = 2,
        limit: int = 10,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Start an asynchronous crawl job.

        Args:
            url: The starting URL for the crawl.
            search_depth: How deep to search (default 1).
            max_depth: Maximum link depth from start URL (default 2).
            limit: Maximum number of pages to crawl (default 10).
            **kwargs: Additional CrawlOptions fields.

        Returns:
            Job ID string on success, None on failure.
        """
        payload: dict[str, Any] = {
            "url": url,
            "searchDepth": search_depth,
            "maxDepth": max_depth,
            "limit": limit,
        }
        payload.update(kwargs)

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.api_url}/v1/crawl",
                json=payload,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict):
                return data.get("jobId") or data.get("job_id")
            return None

        except httpx.HTTPStatusError as e:
            logger.warning("Firecrawl crawl_async HTTP error for %s: %s", url, e.response.status_code)
            return None
        except Exception as e:
            logger.warning("Firecrawl crawl_async failed for %s: %s", url, str(e))
            return None

    async def get_crawl_status(self, job_id: str) -> dict[str, Any]:
        """
        Get the status of a crawl job.

        Args:
            job_id: The job ID returned from crawl_async.

        Returns:
            Dict with job status and data if complete.
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.api_url}/v1/crawl/{job_id}",
                headers=self._build_headers(),
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.warning("Firecrawl crawl status HTTP error for %s: %s", job_id, e.response.status_code)
            return {}
        except Exception as e:
            logger.warning("Firecrawl crawl status failed for %s: %s", job_id, str(e))
            return {}


async def get_firecrawl_client() -> FirecrawlClient:
    """Factory function to create a FirecrawlClient from environment."""
    return FirecrawlClient()