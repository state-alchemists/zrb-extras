#!/usr/bin/env python3
"""
Pytest configuration for zrb-extras tests.
"""

import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Configure asyncio mode for pytest
import pytest

# Enable asyncio mode for all tests
pytest_plugins = ("pytest_asyncio",)

# Configure pytest to run asyncio tests
pytestmark = pytest.mark.asyncio

# Optional: Add any fixtures here


@pytest.fixture
def mock_context():
    """Create a mock context for testing."""
    from unittest.mock import MagicMock

    mock_ctx = MagicMock()
    mock_ctx.print = MagicMock()
    mock_ctx.log_info = MagicMock()
    mock_ctx.log_warning = MagicMock()
    mock_ctx.log_error = MagicMock()
    mock_ctx.is_tty = False

    return mock_ctx
