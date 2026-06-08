"""Utilities for extracting embeddings from text datasets."""

from pathlib import Path

import pandas as pd


def load_sentence_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load a sentence dataset from CSV.

    Args:
        csv_path: Path to CSV file with `label` and `text` columns.

    Returns:
        DataFrame containing labels and text.

    Raises:
        ValueError: If required columns are missing.
    """
    dataset = pd.read_csv(csv_path)
    required_columns = {"label", "text"}

    if not required_columns.issubset(dataset.columns):
        raise ValueError("CSV must contain 'label' and 'text' columns.")

    return dataset
