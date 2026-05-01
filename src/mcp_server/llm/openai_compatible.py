"""
OpenAI-compatible LLM provider implementation.

Supports Ollama, LM Studio, local APIs, and cloud providers with
OpenAI-compatible chat completions interface.
"""

from typing import Optional
import httpx

from src.mcp_server.llm.base import LLMProvider, LLMProviderConfig
from src.mcp_server.llm.exceptions import LLMProviderError


class OpenAICompatibleProvider(LLMProvider):
    """
    LLM provider for OpenAI-compatible API endpoints.
    
    This provider implements the standard OpenAI chat completions format
    and works with:
    - Ollama (localhost:11434)
    - LM Studio
    - Local AI gateways
    - Cloud providers with OpenAI-compatible APIs (OpenRouter, etc.)
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
        """
        Check if the provider endpoint is reachable.
        
        First tries to call /models endpoint, then falls back to a minimal
        chat completions request as health check.
        """
        # Try models endpoint first (lightweight check)
        try:
            base = self._config.base_url.rstrip('/')
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{base}/models")
                if resp.status_code == 200:
                    return True
        except Exception:
            pass

        # Fallback: try a minimal chat completions request
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                await client.post(
                    f"{self._config.base_url}/chat/completions",
                    json={
                        "model": self._config.model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1
                    },
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
                    json={
                        "model": self._config.model,
                        "messages": messages
                    },
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
        """Build request headers including auth if configured."""
        headers = {}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers
