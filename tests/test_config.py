"""Tests for config file parsing."""

from __future__ import annotations

import json

import pytest

from personal_agent.config import load_config
from personal_agent.exceptions import ConfigError


def test_non_mapping_config_file_raises_config_error(tmp_path):
    """A config file whose top level is not an object must raise ConfigError.

    A JSON/YAML file whose top level is a list previously leaked a raw
    TypeError ('Settings() argument after ** must be a mapping, not list'),
    masking the actual config problem.
    """
    path = tmp_path / "config.json"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ConfigError, match="top-level value must be an object"):
        load_config(str(path))


def test_valid_config_file_loads(tmp_path):
    """A well-formed config file still parses to a Settings instance."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"agent": {"provider": "openai"}}))
    settings = load_config(str(path))
    assert settings.agent.provider == "openai"
