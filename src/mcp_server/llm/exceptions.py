"""
Custom exceptions for LLM provider functionality.
"""

from typing import Optional


class LLMProviderError(Exception):
    """
    Raised when an individual LLM provider fails to respond or returns an error.
    
    Attributes:
        provider_name: The identifier of the provider that failed.
        status_code: HTTP status code if available, None otherwise.
    """

    def __init__(
        self,
        provider_name: str,
        message: str,
        status_code: Optional[int] = None
    ):
        self.provider_name = provider_name
        self.status_code = status_code
        super().__init__(f"[{provider_name}] {message}")


class LLMAllProvidersFailedError(Exception):
    """
    Raised when all configured LLM providers fail.
    
    This exception aggregates errors from all failed providers to help
    with debugging and logging.
    """
    pass
