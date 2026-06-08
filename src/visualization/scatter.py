"""Scatter plot utilities for embedding visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_2d_embeddings(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    """Plot 2D embeddings colored by label.

    Args:
        embeddings_2d: 2D embeddings with shape `(num_samples, 2)`.
        labels: Class labels with shape `(num_samples,)`.
        title: Plot title.
        output_path: Path where the figure should be saved.

    Raises:
        ValueError: If embeddings are not 2-dimensional.
    """
    if embeddings_2d.ndim != 2 or embeddings_2d.shape[1] != 2:
        raise ValueError("embeddings_2d must have shape (num_samples, 2).")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(set(labels.tolist()))

    plt.figure(figsize=(7, 5))

    for label in unique_labels:
        label_mask = labels == label
        plt.scatter(
            embeddings_2d[label_mask, 0],
            embeddings_2d[label_mask, 1],
            label=label,
            alpha=0.8,
        )

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
