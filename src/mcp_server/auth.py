import os

from mcp.server.auth.provider import AccessToken, TokenVerifier


class StaticTokenVerifier(TokenVerifier):
    """Verifies Bearer tokens against a static set of pre-shared API keys."""

    def __init__(self, valid_keys: list[str]):
        self._valid_keys = set(valid_keys)

    async def verify_token(self, token: str) -> AccessToken | None:
        if token in self._valid_keys:
            return AccessToken(
                token=token,
                client_id="api-client",
                scopes=["mcp"],
                expires_at=None,
            )
        return None


def load_api_keys_from_env() -> list[str]:
    """Load API keys from the MCP_API_KEYS environment variable.

    Keys are comma-separated and whitespace-trimmed. Returns an empty list
    when the variable is unset or empty, which disables authentication.
    """
    raw = os.getenv("MCP_API_KEYS", "")
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]