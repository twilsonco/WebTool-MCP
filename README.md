# WebTool-MCP

A Model Context Protocol (MCP) server that provides AI assistants with web capabilities including fetching, searching, and summarization.

## Overview

WebTool is an MCP server designed to extend AI assistants with practical web functionality. It enables fetching web pages and converting them to Markdown format, performing multi-provider web searches, and generating LLM-powered summaries of web content.

The server is built with:
- **MCP SDK** for the protocol implementation
- **httpx** for async HTTP requests
- **[Playwright](https://playwright.dev/python/)** for dynamic rendering of JavaScript-heavy pages
- **[Trafilatura](https://trafilatura.readthedocs.io/)** and **[Readability-lxml](https://github.com/buriy/python-readability)** for heuristic text-density extraction
- **BeautifulSoup** and **markdownify** for HTML-to-Markdown conversion
- **[Docling](https://github.com/docling-project/docling)** for layout-aware parsing of PDFs, DOCX, PPTX, XLSX, images, and more
- OpenAI-compatible LLM endpoints for summarization and optional content refinement

## Tools/Functions

### fetch
Fetch URLs and convert content to Markdown format with optional filtering, pagination, and LLM-powered summarization.

Uses a **multi-tiered extraction pipeline** to maximise content quality:

When `USE_FIRECRAWL=true`, Firecrawl is attempted first as the primary extractor:
0. **Firecrawl** (optional) — API-based scraping with JavaScript rendering; if it fails or returns thin content (<200 words), the entire legacy pipeline runs as one fallback block

Legacy pipeline continues with:
1. **Playwright** — renders JavaScript/SPA pages in a headless browser
2. **Trafilatura** — fast heuristic text-density extraction
3. **Readability-lxml** — Mozilla-style article extraction
4. **Docling** — layout-aware parsing for binary documents (PDF, DOCX, PPTX, XLSX, images, CSV, JSON, XML)
5. **BeautifulSoup** — universal HTML fallback (always succeeds)
6. **LLM refinement** — optional semantic cleanup pass (`use_llm_refinement=True`)

Other capabilities:
- Regex-based content filtering with configurable padding
- Word-level truncation and pagination via `start_word` and `num_words`
- Optional extraction of links from fetched pages
- **Summarization** — set `summarize=true` to get an LLM-generated summary instead of raw content

### search
Multi-provider web search with support for:

- **[Miklium](https://github.com/MIKLIUM-Team/MIKLIUM/blob/main/api/search/README.md)** - Free provider, no API key required (always available, but subject to rate limits and potential downtime)
- **[Tavily AI](https://www.tavily.com)** - General-purpose search with good results catered to LLM consumption (requires API key from tavily.com)
- **[Brave Search](https://brave.com/search/api/)** - Privacy-focused search with freshness filters (requires API key from brave.com/search/api)
- **[Google Custom Search](https://developers.google.com/custom-search/docs/overview)** - Programmable Search Engine integration (requires API key and Search Engine ID from Google)

Features include automatic failover between providers and date filtering (results from the last N days).

### agentic_fetch
Autonomous AI-powered web browsing that plans and executes multi-step research tasks.

The agent uses a loop-based approach:
1. Analyze the user's request and decide on an action
2. Execute actions (search, fetch content, navigate pages)
3. Evaluate results and decide next steps
4. Repeat until information is found or max steps reached

**Firecrawl Enhancement:**
When Firecrawl is configured (`FIRECRAWL_API_URL` environment variable), the agent uses it for:
- **Fast URL fetching** — Uses Firecrawl scrape API instead of browser-use for simple content retrieval
- **URL discovery** — Uses `/map` endpoint to discover related URLs during research
- **Batch scraping** — Processes multiple URLs efficiently when needed

The agent falls back through multiple extraction methods:
1. Firecrawl (if configured and available)
2. Browser-use with Playwright (for interactive pages requiring JavaScript rendering)
3. Direct HTTP fetch via extraction pipeline

Example usage:
```python
from mcp_server.agentic import AgenticFetchAgent
from mcp_server.extraction.firecrawl_client import get_firecrawl_client

firecrawl = await get_firecrawl_client()
agent = AgenticFetchAgent(
    llm_manager=llm_manager,
    firecrawl_client=firecrawl,
    max_steps=10
)
result = await agent.execute("Research the latest developments in AI")
```

Or use the convenience function:
```python
from mcp_server.agentic import agentic_fetch

result = await agentic_fetch(
    prompt="Find recent news about Python",
    firecrawl_client=firecrawl,
    max_steps=5
)
```

## Prerequisites

- Python 3.10 or higher
- For **search**: Works out-of-the-box with Miklium (no API key required). Optional: add TAVILY_API_KEY, BRAVE_API_KEY, or GOOGLE_API_KEY + GOOGLE_SEARCH_ENGINE_ID for additional providers
- For **fetch with summarize=true**: An OpenAI-compatible endpoint (Ollama, OpenWebUI, etc.)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd WebTool-MCP
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Create your environment configuration:
```bash
cp .env.example .env
```

4. Configure your API keys in the `.env` file (see Configuration section below).

## Configuration

WebTool uses environment variables for configuration. Copy `.env.example` to `.env` and set the appropriate values.

### LLM Configuration (for fetch summarize)

WebTool supports multiple LLM providers with automatic failover. Providers are tried in order (1, 2, 3...) - if the primary fails, the next provider is used automatically.

**At least one provider must be configured using `LLM_PROVIDER_1_*` variables.**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER_1_NAME` | No | `provider1` | Optional name for Provider 1 (for logging/debugging) |
| `LLM_PROVIDER_1_BASE_URL` | Yes | - | Base URL of Provider 1's OpenAI-compatible API (Ollama, OpenWebUI, etc.) |
| `LLM_PROVIDER_1_MODEL` | No | `llama3.2` | Model name for Provider 1 |
| `LLM_PROVIDER_1_API_KEY` | No | (empty) | API key if Provider 1 requires authentication |
| `LLM_PROVIDER_2_NAME` | No | `provider2` | Optional name for Provider 2 (for logging/debugging) |
| `LLM_PROVIDER_2_BASE_URL` | No* | - | Base URL of Provider 2 (failover) |
| `LLM_PROVIDER_2_MODEL` | No | - | Model name for Provider 2 |
| `LLM_PROVIDER_2_API_KEY` | No | (empty) | API key if Provider 2 requires authentication |

*Only required if configuring failover support.

### Search Provider Configuration (for search)

Web search works out-of-the-box with **Miklium** (no API key required). Additional providers can be configured for redundancy.

#### Miklium (default, no configuration needed)
Free web search provider. No environment variables required - it's always available as the primary provider.

#### Tavily AI
| Variable | Required* | Description |
|----------|-----------|-------------|
| `TAVILY_API_KEY` | Yes* | API key from tavily.com (get one at tavily.com) |

*Only required if you want to use Tavily as a search provider.

#### Brave Search
| Variable | Required* | Description |
|----------|-----------|-------------|
| `BRAVE_API_KEY` | Yes* | API key from brave.com/search/api |

*Only required if you want to use Brave as a search provider.

#### Google Custom Search
| Variable | Required* | Description |
|----------|-----------|-------------|
| `GOOGLE_API_KEY` | Yes* | API key from Google Cloud Console |
| `GOOGLE_SEARCH_ENGINE_ID` | Yes* | Search Engine ID from Programmable Search Engine |

*Only required if you want to use Google as a search provider.

## Usage

### Starting the Server

Run the MCP server (HTTP transport on port 8000):
```bash
uv run python src/mcp_server/server.py
```

Change the host or port with `--host` and `--port`:
```bash
uv run python src/mcp_server/server.py --host 0.0.0.1 --port 9000
```

The server exposes:
- **Health check**: `GET /` → Returns `{"status": "ok", "name": "WebTool MCP Server"}`
- **MCP SSE endpoint**: `GET /mcp` → Establishes an SSE stream for server-to-client messages (required by Roo Code)
- **MCP JSON-RPC endpoint**: `POST /mcp/messages/` → Send JSON-RPC requests to the MCP session

This dual-endpoint design using `mount_sse()` avoids race conditions in HTTP transport's stateless=False mode where GET requests would fail with "Missing session ID" errors.

### Running Examples

The project includes example scripts demonstrating each tool:

```bash
# Run all examples
uv run python examples/run_examples.py all

# Run only fetch examples (includes summarize via summarize=true)
uv run python examples/run_examples.py fetch

# Run only search examples
uv run python examples/run_examples.py search
```

### Tool Reference

#### fetch

Fetch a URL and convert to Markdown.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | Required | URL to fetch |
| `include_links` | bool | `False` | When True, preserve anchor tag hrefs in output; when False (default), unwrap anchor tags keeping only text |
| `start_word` | int | `0` | Starting word index for pagination |
| `num_words` | int | `1000` | Maximum words to return; when summarize=true, this is the max summary word count (passed as LLM constraint) |
| `regex` | str | `None` | Regex pattern to filter content |
| `regex_padding` | int | `50` | Characters of context around regex matches |
| `use_llm_refinement` | bool | `False` | When True, apply an LLM cleanup pass on the extracted Markdown (requires LLM provider configuration) |
| `summarize` | bool | `False` | When true, return an LLM-generated summary instead of raw content (requires LLM provider configuration) |
| `summary_prompt` | str | (empty) | Custom prompt to guide the summarization (optional, uses built-in default if empty) |

**Firecrawl Options** (when USE_FIRECRAWL=true):
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_firecrawl` | bool | `None` | Force use of Firecrawl for this request |
| `screenshot_full_page` | bool | `False` | Capture full page screenshot via Firecrawl |
| `screenshot_quality` | int | `80` | Screenshot quality 1-100 |
| `screenshot_viewport_width` | int | `1920` | Viewport width in pixels for screenshots |
| `screenshot_viewport_height` | int | `1080` | Viewport height in pixels for screenshots |
| `use_clean_content` | bool | `False` | Use onlyCleanContent mode (extract main content only) |
| `extract_schema` | dict | `None` | JSON schema for structured extraction via Firecrawl |
| `extract_prompt` | str | `None` | Prompt guiding JSON extraction |

**Returns:** `{"url": "...", "content": "markdown"}` (normal) or `{"url": "...", "summary": "..."}` when summarize=true; also `{"url": "...", "error": "..."}` on failure

**Example (Python):**
```python
result = await fetch_web_content(
    url="https://example.com",
    num_words=500,
    regex="important|match"
)
```

**Example (curl — HTTP transport):**

The MCP server uses a dual-endpoint design:
- `GET /mcp` — Establishes an SSE stream for server-to-client messages
- `POST /mcp/messages/` — Sends JSON-RPC requests to the session

First initialize a session:
```bash
curl -X POST http://localhost:8000/mcp/messages/ \
  -H "Accept: application/json, text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientInfo":{"name":"curl","version":"1.0"},"protocolVersion":"2025-11-25"}}'
```

Then call the tool (include the `mcp-session-id` header from the initialize response):
```bash
curl -X POST http://localhost:8000/mcp/messages/ \
  -H "Accept: application/json, text/event-stream" \
  -H "Content-Type: application/json" \
  -H "mcp-session-id: <session-id>" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "fetch",
      "arguments": {
        "url": "https://example.com",
        "num_words": 500,
        "regex": "important|match"
      }
    }
  }'
```

#### search

Execute a web search with automatic failover between configured providers.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | Required | Search query string |
| `provider` | str | `"miklium"` | Provider: "tavily", "brave", "google", or "miklium" (default works without API key) |
| `num_results` | int | `10` | Number of results (max 20) |
| `days` | int | `0` | Filter to last N days (0 = no filter). Tavily computes start_date internally; Brave maps to freshness codes (pd/pw/pm/py); Google and Miklium ignore this parameter |
| `offset` | int | `0` | Pagination offset (supported by Brave and Google; not supported by Tavily or Miklium) |

**Returns:**
```python
{
    "query": "search term",
    "provider": "miklium",
    "results": [
        {"title": "Result Title", "url": "https://...", "snippet": "Description"},
        ...
    ],
    "failover_attempts": [...]  # Only present when failover occurred
}
```

**Example (Python):**
```python
result = await search_web(
    query="Python async programming",
    provider="tavily",
    num_results=5
)
```

**Example (curl — HTTP transport):**

After initializing a session (see fetch example above), call the tool:
```bash
curl -X POST http://localhost:8000/mcp/messages/ \
  -H "Accept: application/json, text/event-stream" \
  -H "Content-Type: application/json" \
  -H "mcp-session-id: <session-id>" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "search",
      "arguments": {
        "query": "Python async programming",
        "provider": "tavily",
        "num_results": 5
      }
    }
  }'
```

#### batch_scrape

Batch scrape multiple URLs using Firecrawl. Returns a job ID for status polling.

**Requires:** USE_FIRECRAWL=true in environment configuration.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `urls` | list[str] | Required | List of URLs to scrape |
| `only_main_content` | bool | `True` | Extract only main content (skip navigation, ads, etc.) |

**Returns:** `{"success": true, "job_id": "..."}` or `{"error": "..."}`

**Example (Python):**
```python
result = await api_batch_scrape(
    urls=["https://example.com", "https://example.org"],
    only_main_content=True
)
job_id = result["job_id"]
```

#### batch_status

Poll for batch scrape job status and results from Firecrawl.

**Requires:** USE_FIRECRAWL=true in environment configuration.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `job_id` | str | Required | Job ID returned from batch_scrape |

**Returns:** Job status dict including 'data' field with scraped content when complete

#### map

Discover all URLs on a website without scraping content.

Uses Firecrawl's map endpoint to discover and return URLs found during crawl of the specified website.

**Requires:** USE_FIRECRAWL=true in environment configuration.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | Required | Root URL to start crawling from |

**Returns:**
```python
{"url": "https://example.com", "urls": ["https://example.com/page1", ...], "count": 42}
```

#### screenshot

Capture a screenshot of a URL using Playwright or Firecrawl.

When USE_FIRECRAWL=true, uses Firecrawl for screenshots with enhanced options including full_page capture, quality settings, and viewport dimensions. Otherwise falls back to Playwright-based capture.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | Required | URL to capture |
| `full_page` | bool | `False` | Capture the entire scrollable page (Firecrawl only) |
| `quality` | int | `80` | Image quality 1-100 for JPEG screenshots |
| `viewport_width` | int | `1920` | Viewport width in pixels |
| `viewport_height` | int | `1080` | Viewport height in pixels |

**Returns:** `{"success": true, "image_base64": "...", "url": "..."}` or error dict

## MCP Client Integration

### Claude Desktop

To use WebTool-MCP with Claude Desktop, first start the server (or run it as a persistent service — see "Running as an Always-On Service"), then add the following to your Claude Desktop configuration file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
    "mcpServers": {
        "webtool": {
            "url": "http://localhost:8000/mcp"
        }
    }
}
```

If the server requires Bearer token authentication (when `MCP_API_KEYS` is set), include the token:

```json
{
    "mcpServers": {
        "webtool": {
            "url": "http://localhost:8000/mcp",
            "headers": {
                "Authorization": "Bearer your-api-key"
            }
        }
    }
}
```

Or using process-based communication, without needing to run the server:
```json
{
	"mcpServers": {
		"web tool": {
			"command": "uv",
			"args": [
				"run",
				"--directory",
				"/path/to/WebTool-MCP",
				"webtool-mcp",
				"--stdio"
			]
		}
  }
}
```

After editing the configuration, restart Claude Desktop to load the new MCP server.

### Kilo Code

To use WebTool-MCP with [Kilo Code](https://kilo.ai), add the server configuration to your `kilo.jsonc` file. Kilo Code supports both local (STDIO) and remote (HTTP/SSE) server connections.

**Local (STDIO) — requires the server to be started separately:**

```json
{
  "mcp": {
    "webtool": {
      "type": "local",
      "command": ["uv", "run", "--directory", "/path/to/WebTool-MCP", "python", "src/mcp_server/server.py"],
      "environment": {
        "LLM_PROVIDER_1_BASE_URL": "http://localhost:11434/v1",
        "LLM_PROVIDER_1_MODEL": "llama3.2"
      },
      "enabled": true,
      "timeout": 10000
    }
  }
}
```

*Can also use `.env` file stored in the specified `--directory` for environment variables instead of the `environment` field.*

**Remote (HTTP/SSE) — server must already be running:**

```json
{
  "mcp": {
    "webtool": {
      "type": "remote",
      "url": "http://localhost:8000/mcp",
      "enabled": true,
      "timeout": 15000
    }
  }
}
```

If the server requires Bearer token authentication (when `MCP_API_KEYS` is set), include headers:

```json
{
  "mcp": {
    "webtool": {
      "type": "remote",
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key"
      },
      "enabled": true,
      "timeout": 15000
    }
  }
}
```

Remote servers support OAuth 2.0 authentication automatically if the server offers it; disable with `"oauth": false`.

## Testing

Run the test suite with pytest:

```bash
uv run pytest
```

Run tests with verbose output:

```bash
uv run pytest -v
```

Run only specific test files:

```bash
uv run pytest tests/test_server.py
```

## Running as an Always-On Service

You can run WebTool-MCP as a persistent background service. This is useful when you want the server available at all times without manually starting it each session.

### macOS — launchctl

1. Create a launch agent plist:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.webtool.mcp</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOU/.local/bin/uv</string>
        <string>run</string>
        <string>python</string>
        <string>src/mcp_server/server.py</string>
        <string>--port</string>
        <string>8000</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/YOU/tools/WebTool-MCP</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>LLM_PROVIDER_1_BASE_URL</key>
        <string>http://localhost:11434/v1</string>
        <key>LLM_PROVIDER_1_MODEL</key>
        <string>llama3.2</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/webtool-mcp.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/webtool-mcp.err</string>
</dict>
</plist>
```

2. Save it as `~/Library/LaunchAgents/com.webtool.mcp.plist`, then load and start it:

```bash
launchctl load ~/Library/LaunchAgents/com.webtool.mcp.plist
```

The server will start at login and restart on crash (`KeepAlive`). Adjust the `WorkingDirectory`, `ProgramArguments` path to `uv`, and `EnvironmentVariables` to match your setup.

To stop or unload:

```bash
launchctl unload ~/Library/LaunchAgents/com.webtool.mcp.plist
```

### Linux — systemctl

1. Create a systemd user service unit:

```ini
[Unit]
Description=WebTool MCP Server (HTTP)
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/you/WebTool-MCP
ExecStart=/home/you/.local/bin/uv run python src/mcp_server/server.py --port 8000
Environment=LLM_PROVIDER_1_BASE_URL=http://localhost:11434/v1
Environment=LLM_PROVIDER_1_MODEL=llama3.2
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

2. Save it as `~/.config/systemd/user/webtool-mcp.service`, then enable and start:

```bash
systemctl --user daemon-reload
systemctl --user enable webtool-mcp.service
systemctl --user start webtool-mcp.service
```

To check status or stop:

```bash
systemctl --user status webtool-mcp.service
systemctl --user stop webtool-mcp.service
```

To view logs:

```bash
journalctl --user -u webtool-mcp.service -f
```

### Windows — Task Scheduler

1. Open **Task Scheduler** and create a task under **Task Scheduler Library**.

2. Configure the task:
   - **Triggers:** "At log on" for your user account
   - **Action:** "Start a program"
     - Program: `uv.exe` (full path, e.g. `C:\Users\You\.local\bin\uv.exe`)
     - Arguments: `run python src/mcp_server/server.py --port 8000`
     - Start in: `C:\Users\You\WebTool-MCP`
   - **Conditions:** Uncheck "Start only if the network is available" if you want it to start offline
   - **Settings:** Enable "Restart if the task fails" with a 5-second delay, up to 3 retries

3. Set environment variables by adding a **Environment** tab (available in Task Scheduler via the `Actions` pane → **Properties** → **Settings**), or by defining them system-wide via **System Properties** → **Environment Variables**.

For a more robust service, [NSSM (Non-Sucking Service Manager)](https://nssm.cc/) can wrap the server as a proper Windows service:

```powershell
nssm install WebToolMCP "C:\Users\You\.local\bin\uv.exe" "run python src/mcp_server/server.py --port 8000"
nssm set WebToolMCP AppDirectory "C:\Users\You\WebTool-MCP"
nssm set WebToolMCP AppEnvironmentExtra LLM_PROVIDER_1_BASE_URL=http://localhost:11434/v1 LLM_PROVIDER_1_MODEL=llama3.2
nssm start WebToolMCP
```

## TODO

- **More search sources** — Add academic and preprint search providers: arXiv, ChemRxiv, etc. (pdf link in results can be fetched or summarized to reduce tokens)

## Architecture Notes

### Extraction Pipeline
The server uses a multi-tiered async extraction pipeline for `fetch`. Each tier is tried in order; the first result that meets the minimum quality threshold (≥ 50 words) is returned:

1. **Playwright** — a shared headless Chromium browser (singleton) renders the page with JavaScript, enabling content from SPAs and lazy-loaded sites.
2. **Trafilatura** — applies text-density heuristics to strip boilerplate from the rendered (or raw) HTML.
3. **Readability-lxml** — Mozilla-style article extraction as a secondary heuristic fallback.
4. **Docling** — for binary document URLs (PDF, DOCX, etc.), the content bytes are passed directly to Docling's layout-aware converter.
5. **BeautifulSoup** — universal fallback that always produces output.
6. **LLM refinement** — an optional final pass that sends the extracted Markdown to a configured LLM for semantic cleanup (enabled via `use_llm_refinement=True`).

All HTTP I/O uses `httpx.AsyncClient`.

### API Key Security
All API keys are loaded from the `.env` file at startup. The server never hardcodes credentials and expects them to be provided via environment variables.

### Markdown Output
Content is converted to Markdown format using `markdownify`, making it ideal for consumption by LLMs without HTML parsing overhead.

### Multi-Provider Search
The search tool abstracts over multiple search providers, normalizing their output formats. Miklium is always available as the default provider (no API key required). Additional providers (Tavily, Brave, Google) are enabled when their API keys are configured. When a preferred provider fails or is not configured, the tool automatically fails over to the next available provider in priority order (miklium > tavily > brave > google).

### Brave Freshness Mapping
The `days` parameter is mapped to Brave's freshness codes: 1 day = `pd`, 7 days = `pw`, 31 days = `pm`, 365 days = `py`. Values outside these ranges produce no freshness filter.

## Dependencies

**Core:**
- mcp >= 1.0.0
- fastapi-mcp >= 0.4.0
- httpx >= 0.25.0
- playwright >= 1.40.0
- trafilatura >= 1.8.0
- readability-lxml >= 0.8.1
- beautifulsoup4 >= 4.12.0
- markdownify >= 0.11.0
- docling >= 2.0.0
- python-dotenv >= 1.0.0

**Development:**
- pytest >= 7.4.0
- pytest-asyncio >= 0.21.0
