"""Attention visualization utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    tokens: list[str],
    title: str,
    output_path: str | Path,
) -> None:
    """Plot a single attention head as a heatmap.

    Args:
        attention_matrix: Attention matrix with shape `(num_tokens, num_tokens)`.
        tokens: Token labels.
        title: Plot title.
        output_path: Path where the figure should be saved.

    Raises:
        ValueError: If attention matrix shape does not match token count.
    """
    if attention_matrix.shape != (len(tokens), len(tokens)):
        raise ValueError("Attention matrix shape must match number of tokens.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 7))
    plt.imshow(attention_matrix, aspect="auto")

    plt.xticks(
        ticks=range(len(tokens)),
        labels=tokens,
        rotation=90,
    )
    plt.yticks(
        ticks=range(len(tokens)),
        labels=tokens,
    )

    plt.xlabel("Key tokens")
    plt.ylabel("Query tokens")
    plt.title(title)
    plt.colorbar(label="Attention weight")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
