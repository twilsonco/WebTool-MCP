# WebTool-MCP

A Model Context Protocol (MCP) server that provides AI assistants with web capabilities including fetching, searching, and summarization.

## Overview

WebTool is an MCP server designed to extend AI assistants with practical web functionality. It enables fetching web pages and converting them to Markdown format, performing multi-provider web searches, and generating LLM-powered summaries of web content.

The server is built with:
- **MCP SDK** for the protocol implementation
- **httpx** for async HTTP requests (no browser automation)
- **BeautifulSoup** and **markdownify** for HTML-to-Markdown conversion
- OpenAI-compatible LLM endpoints for summarization

## Tools/Functions

### fetchWebContent
Fetch URLs and convert HTML content to Markdown format with optional filtering and pagination.

- Convert HTML pages to clean Markdown suitable for LLMs
- Regex-based content filtering with configurable padding
- Word-level truncation and pagination via `start_word` and `num_words`
- Optional extraction of links from fetched pages

### searchWeb
Multi-provider web search with support for:

- **[Miklium](https://github.com/MIKLIUM-Team/MIKLIUM/blob/main/api/search/README.md)** - Free provider, no API key required (always available, but subject to rate limits and potential downtime)
- **[Tavily AI](https://www.tavily.com)** - General-purpose search with good results catered to LLM consumption (requires API key from tavily.com)
- **[Brave Search](https://brave.com/search/api/)** - Privacy-focused search with freshness filters (requires API key from brave.com/search/api)
- **[Google Custom Search](https://developers.google.com/custom-search/docs/overview)** - Programmable Search Engine integration (requires API key and Search Engine ID from Google)

Features include automatic failover between providers and date filtering (results from the last N days).

### summarizeWebContent
Fetch a URL and generate an AI-powered summary using a configured LLM endpoint. This way you get the most relevant information without overwhelming the model with too much content.

- Summarization via OpenAI-compatible API
- Configurable prompts for guiding the summary focus
- Automatic truncation of long content before processing

## Prerequisites

- Python 3.10 or higher
- For **searchWeb**: Works out-of-the-box with Miklium (no API key required). Optional: add TAVILY_API_KEY, BRAVE_API_KEY, or GOOGLE_API_KEY + GOOGLE_SEARCH_ENGINE_ID for additional providers
- For **summarizeWebContent**: An OpenAI-compatible endpoint (Ollama, OpenWebUI, etc.)

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

### LLM Configuration (for summarizeWebContent)

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

### Search Provider Configuration (for searchWeb)

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

The server exposes a health check at `/` and the MCP streamable-http endpoint at `/mcp`.

### Running Examples

The project includes example scripts demonstrating each tool:

```bash
# Run all examples
uv run python examples/run_examples.py all

# Run only fetchWebContent examples
uv run python examples/run_examples.py fetch

# Run only searchWeb examples
uv run python examples/run_examples.py search

# Run only summarizeWebContent examples
uv run python examples/run_examples.py summarize
```

### Tool Reference

#### fetchWebContent

Fetch a URL and convert to Markdown.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | Required | URL to fetch |
| `include_links` | bool | `False` | When True, preserve anchor tag hrefs in output; when False (default), unwrap anchor tags keeping only text |
| `start_word` | int | `0` | Starting word index for pagination |
| `num_words` | int | `1000` | Maximum words to return |
| `regex` | str | `None` | Regex pattern to filter content |
| `regex_padding` | int | `50` | Characters of context around regex matches |

**Returns:** `{"url": "...", "content": "markdown"}` or `{"url": "...", "error": "..."}`

**Example (Python):**
```python
result = await web_fetch(
    url="https://example.com",
    num_words=500,
    regex="important|match"
)
```

**Example (curl — HTTP transport):**

First initialize a session:
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Accept: application/json, text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientInfo":{"name":"curl","version":"1.0"},"protocolVersion":"2025-11-25"}}'
```

Then call the tool (include the `mcp-session-id` header from the initialize response):
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Accept: application/json, text/event-stream" \
  -H "Content-Type: application/json" \
  -H "mcp-session-id: <session-id>" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "fetchWebContent",
      "arguments": {
        "url": "https://example.com",
        "num_words": 500,
        "regex": "important|match"
      }
    }
  }'
```

#### searchWeb

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
result = await web_search(
    query="Python async programming",
    provider="tavily",
    num_results=5
)
```

**Example (curl — HTTP transport):**

After initializing a session (see fetchWebContent example above), call the tool:
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Accept: application/json, text/event-stream" \
  -H "Content-Type: application/json" \
  -H "mcp-session-id: <session-id>" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "searchWeb",
      "arguments": {
        "query": "Python async programming",
        "provider": "tavily",
        "num_results": 5
      }
    }
  }'
```

#### summarizeWebContent

Fetch a URL and generate an LLM-powered summary.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | Required | URL to summarize |
| `summary_prompt` | str | (built-in) | Custom prompt for summarization |
| `max_words_per_url` | int | `800` | Max words before truncation |

**Returns:**
```python
{
    "url": "https://example.com",
    "summary": "Summarized content..."
}
```
Or on error:
```python
{
    "url": "https://example.com",
    "error": "Error description..."
}
```

**Example (Python):**
```python
result = await summarizeWebContent(
    url="https://docs.python.org/3/library/asyncio.html",
    summary_prompt="Focus on async/await patterns and best practices.",
    max_words_per_url=1000
)
```

**Example (curl — HTTP transport):**

After initializing a session (see fetchWebContent example above), call the tool:
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Accept: application/json, text/event-stream" \
  -H "Content-Type: application/json" \
  -H "mcp-session-id: <session-id>" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "summarizeWebContent",
      "arguments": {
        "url": "https://docs.python.org/3/library/asyncio.html",
        "summary_prompt": "Focus on async/await patterns and best practices.",
        "max_words_per_url": 1000
      }
    }
  }'
```

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

After editing the configuration, restart Claude Desktop to load the new MCP server.

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
- **AI-powered scraping** — Add browser automation capabilities via browser-use, Playwright, Skyvern, etc. for sites that require JavaScript rendering or interactive navigation.

## Architecture Notes

### Async-Only Design
The server uses `httpx.AsyncClient` for all HTTP operations. There is no browser automation or JavaScript rendering; pages are fetched via direct HTTP requests and parsed with BeautifulSoup.

### API Key Security
All API keys are loaded from the `.env` file at startup. The server never hardcodes credentials and expects them to be provided via environment variables.

### Markdown Output
Content is converted to Markdown format using `markdownify`, making it ideal for consumption by LLMs without HTML parsing overhead.

### Multi-Provider Search
The searchWeb tool abstracts over multiple search providers, normalizing their output formats. Miklium is always available as the default provider (no API key required). Additional providers (Tavily, Brave, Google) are enabled when their API keys are configured. When a preferred provider fails or is not configured, the tool automatically fails over to the next available provider in priority order (miklium > tavily > brave > google).

### Brave Freshness Mapping
The `days` parameter is mapped to Brave's freshness codes: 1 day = `pd`, 7 days = `pw`, 31 days = `pm`, 365 days = `py`. Values outside these ranges produce no freshness filter.

## Dependencies

**Core:**
- mcp >= 1.0.0
- fastapi-mcp >= 0.4.0
- httpx >= 0.25.0
- beautifulsoup4 >= 4.12.0
- markdownify >= 0.11.0
- python-dotenv >= 1.0.0

**Development:**
- pytest >= 7.4.0
- pytest-asyncio >= 0.21.0
