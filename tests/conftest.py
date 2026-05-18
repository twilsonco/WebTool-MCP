import os
import sys
import pytest

# Ensure both the project root and src/ are on sys.path so that both
# `src.mcp_server.*` (used by test imports) and `mcp_server.*` (used by
# internal server imports) resolve to the same source tree regardless of
# which virtual-environment is currently active.
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

pytest_plugins = ["pytest_asyncio"]
