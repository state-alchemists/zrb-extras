#!/usr/bin/env python3
"""
Pytest configuration for zrb-extras tests.
"""

import os
import sys
import pytest

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Configure asyncio mode for pytest

# Enable asyncio mode for all tests
pytest_plugins = ("pytest_asyncio",)

# Configure pytest to run asyncio tests
pytestmark = pytest.mark.asyncio
