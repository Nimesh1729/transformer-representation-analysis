"""Tests for dataset embedding extraction utilities."""

from pathlib import Path

import pandas as pd
import pytest

from src.extraction.dataset_embeddings import load_sentence_dataset


def test_load_sentence_dataset_reads_valid_csv(tmp_path: Path) -> None:
    """Test that a valid sentence dataset CSV is loaded correctly."""
    csv_path = tmp_path / "sentences.csv"

    dataset = pd.DataFrame(
        {
            "label": ["star", "galaxy"],
            "text": [
                "The star emits visible light.",
                "The galaxy contains billions of stars.",
            ],
        }
    )
    dataset.to_csv(csv_path, index=False)

    loaded_dataset = load_sentence_dataset(csv_path)

    assert len(loaded_dataset) == 2
    assert list(loaded_dataset.columns) == ["label", "text"]
    assert loaded_dataset.loc[0, "label"] == "star"
    assert loaded_dataset.loc[1, "label"] == "galaxy"


def test_load_sentence_dataset_raises_for_missing_label_column(
    tmp_path: Path,
) -> None:
    """Test that missing label column raises ValueError."""
    csv_path = tmp_path / "sentences.csv"

    dataset = pd.DataFrame(
        {
            "text": ["The star emits visible light."],
        }
    )
    dataset.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_sentence_dataset(csv_path)


def test_load_sentence_dataset_raises_for_missing_text_column(
    tmp_path: Path,
) -> None:
    """Test that missing text column raises ValueError."""
    csv_path = tmp_path / "sentences.csv"

    dataset = pd.DataFrame(
        {
            "label": ["star"],
        }
    )
    dataset.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_sentence_dataset(csv_path)
