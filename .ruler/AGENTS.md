# WebTool-MCP AI Agent Instructions

## Project Purpose
- WebTool-MCP is a Model Context Protocol (MCP) server exposing web fetch, web search tools. Summarization is now part of `fetchWebContent` via the `summarize=true` parameter.
- The server is built with `FastMCP`, `httpx`, `BeautifulSoup`, `markdownify`, and OpenAI-compatible LLM endpoints.
- The project prioritizes maximal quality output over minimal dependencies, using a multi-tiered extraction pipeline.

## Hard Rules
- Always maintain asynchronous architecture using `httpx`.
- Use `markdownify` for all HTML-to-Markdown conversions.
- API keys must be loaded via `python-dotenv` from `.env`; never hardcode credentials.
- All tools must strictly adhere to the defined `FastMCP` schema.
- Prioritize clean, structured Markdown output for all LLM-facing data.
- Use browser automation (Playwright) when quality of content extraction is paramount.

## Authority & Links
- Project: `WebTool` MCP Server
- Local Infrastructure: Open WebUI (OpenAI-compatible) endpoint
- Dependencies: `mcp`, `httpx`, `beautifulsoup4`, `markdownify`, `python-dotenv`, `playwright`, `trafilatura`, `readability-lxml`

## Setup / Test
- `uv sync`
- `cp .env.example .env` (Add your API keys)
- `uv run python src/mcp_server/server.py`

## Workflow
- `grep -r "TODO" src/` (Check pending implementations)
- `uv run python src/mcp_server/server.py` (Start MCP server)
- `uv run mcp-inspector src/mcp_server/server.py` (Validate tool schemas)
- `uv run pytest tests/` (Run suite)

## Stop Conditions
- Refuse any request to hardcode API credentials in `server.py`.
- Ask for clarification if a new search provider requires a non-standard authentication flow.
- Ask for confirmation before modifying tool input/output schemas.

## Core Rules for Coding Agents
- Use Python 3.10+ idioms: type hints, `async`/`await`, `dataclasses`, and clear module imports.
- All outbound HTTP I/O must be async and use `httpx.AsyncClient` with `async with`.
- Do not hardcode API keys, secrets, or provider credentials anywhere in code.
- Load configuration from `.env` using `dotenv.load_dotenv()` only.

## Code Structure and Style
- `src/mcp_server/server.py` is the entrypoint and MCP tool registry.
- Register tools using `@mcp.tool()` and keep tool implementations focused and testable.
- Place reusable logic, provider abstractions, and failover behavior under `src/mcp_server/llm/`.
- Keep tool outputs JSON-serializable: dictionaries, lists, strings, numbers, booleans.
- Prefer explicit error handling and inline error objects instead of allowing uncaught exceptions to bubble out.
- Keep docstrings concise, descriptive, and aligned with existing triple-quoted module docstrings.

## Project-Specific Behavior
- `fetchWebContent` uses a multi-tiered extraction pipeline for optimal content quality:
  - Dynamic Rendering Layer (Playwright) for SPA/client-side content
  - Heuristic/Text-Density Layer (Trafilatura/Readability-lxml) for fast boilerplate removal
  - Layout-Aware Layer (Docling) for complex document structures
  - Cognitive Extraction Layer (LLM) for optional semantic refinement
- `search_web` supports multiple providers with dynamic provider configuration via environment variables.
  - Provider order and availability are determined by: `miklium` (always enabled), `TAVILY_API_KEY`, `BRAVE_API_KEY`, `GOOGLE_API_KEY` + `GOOGLE_SEARCH_ENGINE_ID`.
  - Search functions should gracefully fail over to the next available provider and include `failover_attempts` when appropriate.
- `fetchWebContent` with summarize=true uses `LLMManager` to perform multi-provider failover across OpenAI-compatible endpoints.
- `LLMManager` should load provider configs from `LLM_PROVIDER_{N}_*` environment variables and preserve priority order.

## Content Extraction Pipeline

The project employs a multi-tiered content extraction pipeline to maximize output quality:

1. **Dynamic Rendering Layer (Playwright)** - Handles SPAs and client-side rendered content by executing JavaScript and waiting for dynamic content to load. Use this layer first when dealing with modern web applications.

2. **Heuristic/Text-Density Layer (Trafilatura/Readability-lxml)** - Provides fast boilerplate removal using text-density algorithms. Ideal for traditional HTML pages with clear content structure.

3. **Layout-Aware Layer (Docling)** - Processes complex document structures including PDFs, tables, and multi-column layouts. Integrates existing Docling infrastructure for advanced document understanding.

4. **Cognitive Extraction Layer (LLM)** - Optional semantic refinement using LLM analysis to improve content quality, structure extraction, and context understanding.

### Pipeline Fallback Strategy
- Start with the simplest effective layer (heuristic-based)
- Escalate to more sophisticated layers only when needed
- Each layer should provide clear quality metrics for downstream selection
- LLM refinement is optional and can be disabled via configuration

## Testing Expectations
- Use `pytest` and `pytest-asyncio` for async tests.
- Mock `httpx.AsyncClient` and `os.getenv` for deterministic external behavior.
- Cover success cases, error handling, no-API-key conditions, and fallback logic.
- Keep tests in `tests/` and follow existing naming conventions: `test_*.py` and `test_*` functions.

## Dependency and Configuration Notes
- Keep dependencies aligned with `pyproject.toml`.
- Browser automation (Playwright) and content extraction libraries (Trafilatura, Readability-lxml) are first-class dependencies for quality-first extraction.
- Use environment variables for provider configuration, not hardcoded values.
- Preserve the default `User-Agent` header pattern used in `server.py` when making web requests.

## Ruler Instructions
- This file is the main agent instruction entrypoint for this repository.
- Keep guidance project-specific and practical; avoid generic Python rules unless they reflect current repository style.
- When adding new features, align with the existing async, environment-driven design and quality-first philosophy.
