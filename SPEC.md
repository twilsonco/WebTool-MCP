# Multi-Provider LLM Failover System Specification

## Overview

This document specifies the design for a multi-provider LLM (Large Language Model) failover system that allows the WebTool-MCP server to use multiple LLM providers with automatic failover when a provider fails.

---

## 1. Environment Variable Naming Convention

### Multi-Provider Configuration
Providers are numbered sequentially starting from 1:

```bash
# Provider 1 (highest priority)
LLM_PROVIDER_1_NAME=ollama-local
LLM_PROVIDER_1_BASE_URL=http://localhost:11434/v1
LLM_PROVIDER_1_API_KEY=
LLM_PROVIDER_1_MODEL=llama3.2

# Provider 2 (failover)
LLM_PROVIDER_2_NAME=openai-cloud
LLM_PROVIDER_2_BASE_URL=https://api.openai.com/v1
LLM_PROVIDER_2_API_KEY=sk-...
LLM_PROVIDER_2_MODEL=gpt-4o-mini

# Provider 3 (fallback)
LLM_PROVIDER_3_NAME=anthropic-cloud
LLM_PROVIDER_3_BASE_URL=https://api.anthropic.com/v1
LLM_PROVIDER_3_API_KEY=sk-ant-...
LLM_PROVIDER_3_MODEL=claude-3-haiku-20240307
```

### Environment Variable Rules
| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER_{N}_NAME` | Yes | Unique identifier for the provider |
| `LLM_PROVIDER_{N}_BASE_URL` | Yes | OpenAI-compatible API base URL |
| `LLM_PROVIDER_{N}_API_KEY` | No | Authentication token (empty if not needed) |
| `LLM_PROVIDER_{N}_MODEL` | Yes | Model identifier for chat completions |

---

## 2. Provider Configuration Structure

### dataclass: LLMProviderConfig
```python
@dataclass(frozen=True)
class LLMProviderConfig:
    name: str                           # Unique provider identifier
    base_url: str                       # API base URL (e.g., "http://localhost:11434/v1")
    api_key: str                        # Authentication token (empty string if not needed)
    model: str                          # Model name for chat completions
```

---

## 3. Abstract LLMProvider Interface

### ABC Class: LLMProvider
```python
from abc import ABC, abstractmethod
from typing import Optional
import httpx

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider's unique identifier."""
        pass

    @property
    @abstractmethod
    def config(self) -> LLMProviderConfig:
        """Return the provider configuration."""
        pass

    @abstractmethod
    async def is_available(self, timeout: float = 5.0) -> bool:
        """
        Check if the provider endpoint is reachable and responsive.
        
        Args:
            timeout: Maximum seconds to wait for a response.
            
        Returns:
            True if provider is available, False otherwise.
        """
        pass

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Send a completion request to the LLM.
        
        Args:
            prompt: The user message content.
            system_prompt: Optional system message for context.
            
        Returns:
            The assistant's response content as a string.
            
        Raises:
            LLMProviderError: If the request fails for any reason.
        """
        pass
```

### Exception Class: LLMProviderError
```python
class LLMProviderError(Exception):
    """Raised when an LLM provider fails to respond or returns an error."""
    
    def __init__(self, provider_name: str, message: str, status_code: Optional[int] = None):
        self.provider_name = provider_name
        self.status_code = status_code
        super().__init__(f"[{provider_name}] {message}")
```

### Concrete Implementation: OpenAICompatibleProvider
```python
class OpenAICompatibleProvider(LLMProvider):
    """
    LLM provider for OpenAI-compatible API endpoints.
    Supports Ollama, LM Studio, local APIs, and cloud providers with OpenAI-compatible interfaces.
    """

    def __init__(self, config: LLMProviderConfig):
        self._config = config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> LLMProviderConfig:
        return self._config

    async def is_available(self, timeout: float = 5.0) -> bool:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{self._config.base_url.rstrip('/v1')}/models")
                return resp.status_code == 200
        except Exception:
            # Fallback: try a simple health check via chat completions
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    await client.post(
                        f"{self._config.base_url}/chat/completions",
                        json={"model": self._config.model, "messages": [{"role": "user", "content": "hi"}]},
                        headers=self._headers()
                    )
                return True
            except Exception:
                return False

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = self._headers()
        headers["Content-Type"] = "application/json"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self._config.base_url}/chat/completions",
                    json={"model": self._config.model, "messages": messages},
                    headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise LLMProviderError(
                self._config.name,
                f"API error {e.response.status_code}: {e.response.text[:200]}",
                status_code=e.response.status_code
            )
        except Exception as e:
            raise LLMProviderError(self._config.name, f"Inference failed: {str(e)}")

    def _headers(self) -> dict:
        headers = {}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers
```

---

## 4. LLMManager Class Design

### Class: LLMManager
```python
from typing import Optional, List
import os

class LLMManager:
    """
    Manages multiple LLM providers with failover support.
    
    Loads provider configurations from environment variables and attempts
    calls in priority order until one succeeds.
    """

    def __init__(self):
        self._providers: List[LLMProvider] = []
        self._load_providers()

    def _load_providers(self) -> None:
        """
        Load provider configurations from environment variables.
        
        Loads multi-provider configuration (LLM_PROVIDER_{N}_*) and creates
        providers in priority order.
        
        Raises:
            ValueError: If a provider is missing required BASE_URL or MODEL.
        """
        n = 1
        while True:
            name = os.getenv(f"LLM_PROVIDER_{n}_NAME")
            if not name:
                break
            
            base_url = os.getenv(f"LLM_PROVIDER_{n}_BASE_URL", "").rstrip('/')
            api_key = os.getenv(f"LLM_PROVIDER_{n}_API_KEY", "")
            model = os.getenv(f"LLM_PROVIDER_{n}_MODEL", "")
            
            if not base_url or not model:
                raise ValueError(
                    f"Provider '{name}' is missing required configuration: "
                    f"BASE_URL={base_url}, MODEL={model}"
                )
            
            config = LLMProviderConfig(
                name=name,
                base_url=base_url,
                api_key=api_key,
                model=model
            )
            self._providers.append(OpenAICompatibleProvider(config))
            n += 1

    @property
    def providers(self) -> List[LLMProvider]:
        """Return the list of configured providers in priority order."""
        return self._providers.copy()

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Attempt completion using providers in failover order.
        
        Args:
            prompt: The user message content.
            system_prompt: Optional system message for context.
            
        Returns:
            The assistant's response content as a string.
            
        Raises:
            LLMAllProvidersFailedError: If all providers fail.
        """
        errors = []
        
        for provider in self._providers:
            try:
                return await provider.complete(prompt, system_prompt)
            except LLMProviderError as e:
                errors.append(str(e))
                continue
        
        raise LLMAllProvidersFailedError(
            f"All {len(self._providers)} LLM providers failed. "
            f"Errors: {'; '.join(errors)}"
        )
```

### Exception Class: LLMAllProvidersFailedError
```python
class LLMAllProvidersFailedError(Exception):
    """Raised when all configured LLM providers fail."""
    pass
```

---

## 5. Failover Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        User Request                         │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
              ┌─────────────────────────────┐
              │     LLMManager.complete()   │
              └──────────────┬──────────────┘
                             ▼
         ┌────────────────────────────────────┐
         │  Iterate through providers list    │
         │  (in priority order: 1, 2, 3...)   │
         └───────────────┬────────────────────┘
                         ▼
              ┌────────���─────────────┐
              │  Try Provider N      │
              │  complete()          │
              └───────────┬──────────┘
                          ▼
            ┌─────────────────────────┐
            │     Success?            │
            └───────────┬─────────────┘
               Yes      │      No
                │       │      │
                ▼       │      ▼
        ┌──────────┐   │  ┌────────────────────┐
        │ Return   │   │  │ Log error          │
        │ response │   │  │ Mark provider as   │
        └──────────┘   │  │ failed             │
                      │  └─────────┬──────────┘
                      │            ▼
                      │  ┌─────────────────────┐
                      │  │ More providers      │
                      │  │ remaining?          │
                      │  └──────────┬──────────┘
                      │    Yes      │     No
                      │     │       │      │
                      │     ▼       │      ▼
                      │  ┌─────┐   │  ┌───────────────────┐
                      └──│Next │◄──┘  │ LLMAllProviders   │
                         └─────┘      │ FailedError       │
                                    └───────────────────┘
```

---

## 6. File Structure for New Code

```
src/mcp_server/
├── __init__.py              # No changes (or add exports)
├── server.py                # Modified: replace _call_llm() with LLMManager
└── llm/                     # NEW: LLM module directory
    ├── __init__.py          # Exports: LLMProvider, LLMManager, exceptions
    ├── base.py              # Abstract LLMProvider ABC
    ├── openai_compatible.py # OpenAICompatibleProvider implementation
    ├── manager.py           # LLMManager class with failover logic
    └── exceptions.py        # Custom exception classes
```

---

## 7. server.py Modification Specification

### Changes Required

1. **Add imports** (after existing imports):
   ```python
   from src.mcp_server.llm import LLMManager, LLMAllProvidersFailedError
   ```

2. **Replace global configuration block**:
   ```python
   # REMOVE:
   # BASE_URL = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:11434/v1")
   # MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.2")
   # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

   # ADD:
   llm_manager = LLMManager()  # Initialized at module load with multi-provider support
   ```

   The `LLMManager` automatically discovers and configures providers from environment variables:
   - `LLM_PROVIDER_1_BASE_URL`, `LLM_PROVIDER_1_MODEL`, `LLM_PROVIDER_1_API_KEY` (required for first provider)
   - `LLM_PROVIDER_2_*`, `LLM_PROVIDER_3_*`, etc. (optional, for failover)
   - Providers are tried in order; if one fails, the next is used automatically

3. **Replace `_call_llm()` function**:
   ```python
   async def _call_llm(prompt: str, system_prompt: Optional[str] = None) -> str:
       """
       Call the configured LLM endpoint(s) with failover support.
       Returns the assistant's response content.
       Raises RuntimeError if all providers fail.
       """
       try:
           return await llm_manager.complete(prompt, system_prompt)
       except LLMAllProvidersFailedError as e:
           raise RuntimeError(str(e))
   ```

4. **No changes to `web_summarize()` function signature or behavior** - it continues to call `_call_llm()` internally.

---

## 8. Adding New Provider Types (Future Extension)

## 9. Adding New Provider Types (Future Extension)

To add support for non-OpenAI-compatible providers (e.g., direct Anthropic API):

1. Create new provider class inheriting from `LLMProvider`:
   ```python
   class AnthropicProvider(LLMProvider):
       async def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
           # Use Anthropic's specific API format
           ...
   ```

2. Update `LLMManager._load_providers()` to detect provider type and instantiate appropriate class:
   ```python
   if base_url.includes("anthropic"):
       provider = AnthropicProvider(config)
   else:
       provider = OpenAICompatibleProvider(config)
   ```

3. Document the new provider configuration requirements.

---

## 9. Testing Strategy (Implementation Note)

- Unit test each `LLMProvider` subclass in isolation with mocked HTTP responses
- Integration test `LLMManager.complete()` failover behavior by mocking sequential failures
- Test error aggregation when all providers fail
