import os
import pytest
from unittest.mock import patch

from src.mcp_server.auth import StaticTokenVerifier, load_api_keys_from_env


class TestStaticTokenVerifier:
    @pytest.mark.asyncio
    async def test_valid_key_returns_access_token(self):
        verifier = StaticTokenVerifier(["key1", "key2"])
        result = await verifier.verify_token("key1")
        assert result is not None
        assert result.token == "key1"
        assert result.client_id == "api-client"
        assert result.scopes == ["mcp"]
        assert result.expires_at is None

    @pytest.mark.asyncio
    async def test_invalid_key_returns_none(self):
        verifier = StaticTokenVerifier(["key1", "key2"])
        result = await verifier.verify_token("bad-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_key_set_rejects_everything(self):
        verifier = StaticTokenVerifier([])
        result = await verifier.verify_token("any-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_second_key_valid(self):
        verifier = StaticTokenVerifier(["alpha", "bravo"])
        result = await verifier.verify_token("bravo")
        assert result is not None
        assert result.token == "bravo"


class TestLoadApiKeysFromEnv:
    def test_present_keys(self):
        with patch("src.mcp_server.auth.os.getenv", return_value="key1,key2,key3"):
            keys = load_api_keys_from_env()
        assert keys == ["key1", "key2", "key3"]

    def test_empty_var_returns_empty_list(self):
        with patch("src.mcp_server.auth.os.getenv", return_value=""):
            keys = load_api_keys_from_env()
        assert keys == []

    def test_unset_var_returns_empty_list(self):
        with patch("src.mcp_server.auth.os.getenv", return_value=""):
            keys = load_api_keys_from_env()
        assert keys == []

    def test_whitespace_trimmed(self):
        with patch("src.mcp_server.auth.os.getenv", return_value="  key1 , key2  ,  key3  "):
            keys = load_api_keys_from_env()
        assert keys == ["key1", "key2", "key3"]

    def test_single_key(self):
        with patch("src.mcp_server.auth.os.getenv", return_value="only-one-key"):
            keys = load_api_keys_from_env()
        assert keys == ["only-one-key"]

    def test_empty_entries_filtered(self):
        with patch("src.mcp_server.auth.os.getenv", return_value="key1,, ,key2"):
            keys = load_api_keys_from_env()
        assert keys == ["key1", "key2"]