# WebTool MCP Server Examples

This directory contains example scripts demonstrating how to interact with the WebTool MCP Server.

## Example Scripts

### run_examples.py

Entry-point launcher that runs example scripts by tool name. Supports `fetch`, `search`, or `all` (runs all in sequence with prompts between each).

```bash
uv run python examples/run_examples.py fetch      # fetch demos only (includes summarize via summarize=true)
uv run python examples/run_examples.py search     # search demos only
uv run python examples/run_examples.py all        # everything in sequence
```

### test_mcp.py

End-to-end MCP client test that connects via the streamable-http transport and calls all tools through the MCP protocol (JSON-RPC). Tests `search` with each provider (miklium, tavily, brave, google), and `fetch` (normal fetch and summarize modes).

**How to Run:**

1. Start the MCP server:
   ```bash
   uv run python src/mcp_server/server.py --http
   ```

2. In a separate terminal, run the test script:
   ```bash
   uv run python examples/test_mcp.py
   ```

**Requirements:**
- Server must be running on `http://localhost:8000` (or set `MCP_SERVER_PORT` env var)
- Internet connection for making web requests
- For summarize mode (`fetch` with summarize=true), configure LLM providers in `.env` (optional - search and normal fetch work without it)

### fetch_web_content_examples.py

Direct-call examples importing the actual `fetch` implementation from server.py. Each call fetches a single URL. No API keys needed (except example 7, which requires an LLM provider).

1. **Basic Fetch** - Fetch a URL and display raw Markdown; repeated with `include_links=False`
2. **Word Truncation** - `num_words=50` truncates output to 50 words
3. **Regex Filtering** - `regex="the|is"` with `regex_padding=30` for context around matches
4. **Word Offset** - `start_word=0` vs `start_word=50` for pagination by word position
5. **Binary Document Formats** - Lists extensions routed directly to Docling (PDF, DOCX, PPTX, XLSX, images, CSV, JSON, XML)
6. **PDF Fetch** - Fetches a real PDF via the binary-document path (Docling)
7. **LLM Refinement** - `use_llm_refinement=True` applies an optional LLM semantic cleanup pass (requires `LLM_PROVIDER_1_*` in `.env`)
8. **Full Content Fetch** - Fetches real-world URLs with full content extraction

```bash
uv run python examples/fetch_web_content_examples.py          # run all examples
uv run python examples/fetch_web_content_examples.py 1        # example 1 only
uv run python examples/fetch_web_content_examples.py 1-3,5    # examples 1, 2, 3, and 5
```

**Extraction Pipeline:**
`fetch` uses a multi-tiered pipeline to maximise content quality: Playwright (JS rendering) → Trafilatura → Readability-lxml → Docling (binary documents) → BeautifulSoup (fallback). An optional LLM refinement pass is available via `use_llm_refinement=True`.

### search_web_examples.py

Direct-call examples importing the actual `search` implementation. Each call performs a single search with one provider. Supports a `--dry-run` flag to print search specs without making API calls.

1. **Miklium Search** - Default, free provider (no API key required)
2. **Tavily Search** - Requires `TAVILY_API_KEY`
3. **Brave Search** - Requires `BRAVE_API_KEY`
4. **Google Custom Search** - Requires `GOOGLE_API_KEY` + `GOOGLE_SEARCH_ENGINE_ID`
5. **Date Filtering** - `days=730` (2 years); Brave maps to freshness codes, Tavily computes start_date
6. **Date Options** - Brave-specific: `days=1` (`pd`), `7` (`pw`), `31` (`pm`), `365` (`py`)
7. **Offset Pagination** - `offset=N` for Brave/Google; Tavily does not support offsets
8. **Google Ignores Days** - Shows Google silently drops the `days` parameter
9. **Error Handling** - Unknown provider and empty query produce error objects, not exceptions
10. **Config Check** - Prints which search API keys are SET vs NOT SET

```bash
uv run python examples/search_web_examples.py              # run all 10 examples
uv run python examples/search_web_examples.py 1            # example 1 only (miklium, free)
uv run python examples/search_web_examples.py 1-4          # examples 1–4 (all providers)
uv run python examples/search_web_examples.py 1,9,10       # examples 1, 9, and 10
uv run python examples/search_web_examples.py --dry-run    # print specs, no API calls
```

### summarize_web_content_examples.py (removed)

This file has been removed. Summarization is now part of `fetch` — use it with `summarize=true` and optional `summary_prompt`.

### sse_streaming_example.py

Demonstrates SSE (Server-Sent Events) streaming with WebTool-MCP. Shows how to:
- Connect using the streamable-http transport
- Establish SSE streaming via GET /mcp endpoint
- Send JSON-RPC requests and receive responses

The MCP server uses `mount_sse()` which provides:
- **GET /mcp** → SSE stream for server-to-client messages (Roo Code integration)
- **POST /mcp/messages/** → send JSON-RPC requests to the session

```bash
# Start the MCP server in HTTP mode first:
uv run python src/mcp_server/server.py --http

# In another terminal, run the SSE streaming example:
uv run python examples/sse_streaming_example.py           # all demos
uv run python examples/sse_streaming_example.py search    # only search demo
uv run python examples/sse_streaming_example.py fetch     # only fetch demo
```