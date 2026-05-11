"""
Abstract base classes for LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMProviderConfig:
    """
    Immutable configuration for an LLM provider.
    
    Attributes:
        name: Unique identifier for the provider.
        base_url: OpenAI-compatible API base URL (e.g., "http://localhost:11434/v1").
        api_key: Authentication token. Empty string if not required.
        model: Model identifier for chat completions.
    """
    name: str
    base_url: str
    api_key: str
    model: str


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All concrete provider implementations must inherit from this class
    and implement the required abstract methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:  # pragma: no cover
        """Return the provider's unique identifier."""
        pass

    @property
    @abstractmethod
    def config(self) -> LLMProviderConfig:  # pragma: no cover
        """Return the provider configuration."""
        pass

    @abstractmethod
    async def is_available(self, timeout: float = 5.0) -> bool:  # pragma: no cover
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
    ) -> str:  # pragma: no cover
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
