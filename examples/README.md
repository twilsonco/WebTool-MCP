# WebTool MCP Server Examples

This directory contains example scripts demonstrating how to interact with the WebTool MCP Server.

## Example Scripts

### run_examples.py

Entry-point launcher that runs example scripts by tool name. Supports `fetch`, `search`, `summarize`, or `all` (runs all three in sequence with prompts between each).

```bash
uv run python examples/run_examples.py fetch      # fetchWebContent demos only
uv run python examples/run_examples.py search     # searchWeb demos only
uv run python examples/run_examples.py summarize  # summarizeWebContent demos only
uv run python examples/run_examples.py all        # everything in sequence
```

### test_mcp.py

End-to-end MCP client test that connects via the streamable-http transport and calls all three tools through the MCP protocol (JSON-RPC). Tests `searchWeb` with each provider (miklium, tavily, brave, google), `fetchWebContent`, and `summarizeWebContent`.

**How to Run:**

1. Start the MCP server:
   ```bash
   uv run python src/mcp_server/server.py
   ```

2. In a separate terminal, run the test script:
   ```bash
   uv run python examples/test_mcp.py
   ```

**Requirements:**
- Server must be running on `http://localhost:8000` (or set `MCP_SERVER_PORT` env var)
- Internet connection for making web requests
- For `summarizeWebContent`, configure LLM providers in `.env` (optional - search and fetch work without it)

### fetch_web_content_examples.py

Direct-call examples importing the actual `fetchWebContent` implementation from server.py. Each call fetches a single URL. No API keys needed.

1. **Basic Fetch** - Fetch a URL and display raw Markdown
2. **Word Truncation** - `num_words=50` truncates output to 50 words
3. **Regex Filtering** - `regex="the|is"` with `regex_padding=30` for context around matches
4. **Word Offset** - `start_word=0` vs `start_word=50` for pagination by word position
5. **Docling Document Formats** - Demonstrates multi-format document parsing (PDF, DOCX, PPTX, XLSX, images)
6. **Docling Fallback Behavior** - Shows how the system falls back to BeautifulSoup when Docling is unavailable

```bash
uv run python examples/fetch_web_content_examples.py
```

**Docling Integration:**
- `fetchWebContent` automatically detects and parses supported document formats using Docling
- Supported formats: PDF, DOCX, PPTX, XLSX, PNG, JPG, TIFF, BMP, MD, CSV, JSON, XML, HTML
- Falls back to BeautifulSoup for regular web pages when Docling is not applicable or fails

### search_web_examples.py

Direct-call examples importing the actual `searchWeb` implementation. Each call performs a single search with one provider. Supports a `--dry-run` flag to print search specs without making API calls, and positional provider arguments (e.g., `tavily`, `brave`, `google`).

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
uv run python examples/search_web_examples.py              # all providers
uv run python examples/search_web_examples.py tavily       # Tavily only
uv run python examples/search_web_examples.py --dry-run    # print specs, no API calls
```

### summarize_web_content_examples.py

Direct-call examples importing the actual `summarizeWebContent` implementation. Each call summarizes a single URL. Requires LLM provider configuration.

1. **Single URL** - Summarize one URL with `max_words_per_url=500`
2. **Custom Summary Prompt** - Custom prompt guiding what the LLM extracts
3. **LLM Config Check** - Initializes `LLMManager` and prints provider info (name, base_url, model)
4. **Multi-Provider Failover** - Demonstrates multi-provider LLM failover (commented out in `main()` by default)

```bash
uv run python examples/summarize_web_content_examples.py
```

Note: Examples 2-4 and 6 are commented out in `main()` by default since they make multiple LLM calls. Uncomment them as needed.