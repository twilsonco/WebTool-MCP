# WebTool-MCP AI Agent Instructions

## Project Purpose
- WebTool-MCP is a Model Context Protocol (MCP) server exposing web fetch, web search, and web summarize tools.
- The server is built with `FastMCP`, `httpx`, `BeautifulSoup`, `markdownify`, and OpenAI-compatible LLM endpoints.
- The project prioritizes async architecture, minimal dependencies, and environment-driven configuration.

## Hard Rules
- Always maintain asynchronous architecture using `httpx`.
- Use `markdownify` for all HTML-to-Markdown conversions.
- Never use Selenium or Playwright; keep dependencies minimal and lightweight.
- API keys must be loaded via `python-dotenv` from `.env`; never hardcode credentials.
- All tools must strictly adhere to the defined `FastMCP` schema.
- Prioritize clean, structured Markdown output for all LLM-facing data.

## Authority & Links
- Project: `WebTool` MCP Server
- Local Infrastructure: Open WebUI (OpenAI-compatible) endpoint
- Dependencies: `mcp`, `httpx`, `beautifulsoup4`, `markdownify`, `python-dotenv`

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
- Refuse any request to introduce browser automation (Selenium/Playwright).
- Refuse any request to hardcode API credentials in `server.py`.
- Ask for clarification if a new search provider requires a non-standard authentication flow.
- Ask for confirmation before modifying tool input/output schemas.

## Core Rules for Coding Agents
- Use Python 3.10+ idioms: type hints, `async`/`await`, `dataclasses`, and clear module imports.
- All outbound HTTP I/O must be async and use `httpx.AsyncClient` with `async with`.
- Do not add browser automation dependencies such as Selenium or Playwright.
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
- `web_fetch` converts HTML to Markdown using `BeautifulSoup` + `markdownify` and supports regex filtering, padding, pagination, and link extraction.
- `web_search` supports multiple providers with dynamic provider configuration via environment variables.
  - Provider order and availability are determined by: `miklium` (always enabled), `TAVILY_API_KEY`, `BRAVE_API_KEY`, `GOOGLE_API_KEY` + `GOOGLE_SEARCH_ENGINE_ID`.
  - Search functions should gracefully fail over to the next available provider and include `failover_attempts` when appropriate.
- `web_summarize` uses `LLMManager` to perform multi-provider failover across OpenAI-compatible endpoints.
- `LLMManager` should load provider configs from `LLM_PROVIDER_{N}_*` environment variables and preserve priority order.

## Testing Expectations
- Use `pytest` and `pytest-asyncio` for async tests.
- Mock `httpx.AsyncClient` and `os.getenv` for deterministic external behavior.
- Cover success cases, error handling, no-API-key conditions, and fallback logic.
- Keep tests in `tests/` and follow existing naming conventions: `test_*.py` and `test_*` functions.

## Dependency and Configuration Notes
- Keep dependencies minimal and aligned with `pyproject.toml`.
- No browser automation or heavyweight UI frameworks.
- Use environment variables for provider configuration, not hardcoded values.
- Preserve the default `User-Agent` header pattern used in `server.py` when making web requests.

## Ruler Instructions
- This file is the main agent instruction entrypoint for this repository.
- Keep guidance project-specific and practical; avoid generic Python rules unless they reflect current repository style.
- When adding new features, align with the existing async, environment-driven design and keep implementation lightweight.
