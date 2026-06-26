"""Shared test fixtures for memory subsystem tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_memory_dir():
    """Create a temporary directory for FileMemoryStore tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)