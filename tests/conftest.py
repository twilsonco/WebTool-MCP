import os
import sys
import pytest

# Ensure src is in path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest_plugins = ["pytest_asyncio"]
