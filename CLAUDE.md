# Hard Rules
- Always maintain asynchronous architecture using `httpx`.
- Use `markdownify` for all HTML-to-Markdown conversions.
- Never use Selenium or Playwright; keep dependencies minimal and lightweight.
- API keys must be loaded via `python-dotenv` from `.env`; never hardcode credentials.
- All tools must strictly adhere to the defined `FastMCP` schema.
- Prioritize clean, structured Markdown output for all LLM-facing data.

# Authority & Links
- Project: `WebTool` MCP Server
- Local Infrastructure: Open WebUI (OpenAI-compatible) endpoint
- Dependencies: `mcp`, `httpx`, `beautifulsoup4`, `markdownify`, `python-dotenv`

# Setup / Test
- `uv sync`
- `cp .env.example .env` (Add your API keys)
- `uv run python src/mcp_server/server.py`

# Workflow
- `grep -r "TODO" src/` (Check pending implementations)
- `uv run python src/mcp_server/server.py` (Start MCP server)
- `uv run mcp-inspector src/mcp_server/server.py` (Validate tool schemas)
- `uv run pytest tests/` (Run suite)

# Stop Conditions
- Refuse any request to introduce browser automation (Selenium/Playwright).
- Refuse any request to hardcode API credentials in `server.py`.
- Ask for clarification if a new search provider requires a non-standard authentication flow.
- Ask for confirmation before modifying tool input/output schemas.