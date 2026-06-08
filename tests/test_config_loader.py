"""Tests for configuration loading utilities."""

from pathlib import Path

import pytest
import yaml

from src.utils.config_loader import load_config


def test_load_config_reads_yaml_file(tmp_path: Path) -> None:
    """Test that a valid YAML config file is loaded."""
    config_path = tmp_path / "config.yaml"
    config_data = {
        "model": {"name": "distilbert-base-uncased"},
        "system": {"seed": 42, "device": "cpu"},
    }

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config_data, file)

    loaded_config = load_config(config_path)

    assert loaded_config["model"]["name"] == "distilbert-base-uncased"
    assert loaded_config["system"]["seed"] == 42
    assert loaded_config["system"]["device"] == "cpu"


def test_load_config_raises_for_missing_file() -> None:
    """Test that missing config files raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("missing_config.yaml")


def test_load_config_raises_for_empty_file(tmp_path: Path) -> None:
    """Test that empty config files raise ValueError."""
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(config_path)
