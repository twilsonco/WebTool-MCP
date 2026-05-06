"""
LLM Manager with multi-provider failover support.

Loads provider configurations from environment variables and attempts
calls in priority order until one succeeds.
"""

import os
from typing import List, Optional

from .base import LLMProvider, LLMProviderConfig
from .exceptions import LLMAllProvidersFailedError, LLMProviderError
from .openai_compatible import OpenAICompatibleProvider


class LLMManager:
    """
    Manages multiple LLM providers with failover support.
    
    Loads provider configurations from environment variables and attempts
    calls in priority order until one succeeds.
    
    Configuration is loaded on initialization. Requires multi-provider
    configuration (LLM_PROVIDER_{N}_*) to be set.
    """

    def __init__(self):
        self._providers: List[LLMProvider] = []
        self._load_providers()

    def _load_providers(self) -> None:
        """
        Load provider configurations from environment variables.
        
        Loads multi-provider configuration (LLM_PROVIDER_1_*, LLM_PROVIDER_2_*, etc.)
        and creates providers in priority order.
        
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

            if not base_url:
                raise ValueError(
                    f"Provider '{name}' is missing required LLM_PROVIDER_{n}_BASE_URL"
                )
            if not model:
                raise ValueError(
                    f"Provider '{name}' is missing required LLM_PROVIDER_{n}_MODEL"
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
        """
        Return the list of configured providers in priority order.
        
        Returns a copy to prevent external modification.
        """
        return self._providers.copy()

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Attempt completion using providers in failover order.
        
        Tries each provider sequentially. If a provider fails, logs the error
        and attempts the next one. Returns on first successful response.
        
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
