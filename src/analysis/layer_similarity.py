"""Layer-wise similarity analysis utilities."""

import numpy as np


def compute_layer_self_similarity(
    layer_a: np.ndarray,
    layer_b: np.ndarray,
) -> float:
    """Compute average same-sample cosine similarity between two layers.

    Args:
        layer_a: Embeddings from one layer with shape (num_samples, hidden_size).
        layer_b: Embeddings from another layer with shape (num_samples, hidden_size).

    Returns:
        Average cosine similarity between matching samples.
    """
    normalized_a = layer_a / np.linalg.norm(layer_a, axis=1, keepdims=True)
    normalized_b = layer_b / np.linalg.norm(layer_b, axis=1, keepdims=True)

    similarities = np.sum(normalized_a * normalized_b, axis=1)

    return float(similarities.mean())


def compute_layer_similarity_matrix(
    layer_embeddings: list[np.ndarray],
) -> np.ndarray:
    """Compute pairwise layer similarity matrix.

    Args:
        layer_embeddings: List of layer embedding matrices.

    Returns:
        Matrix with shape (num_layers, num_layers).
    """
    num_layers = len(layer_embeddings)
    similarity_matrix = np.zeros((num_layers, num_layers))

    for i in range(num_layers):
        for j in range(num_layers):
            similarity_matrix[i, j] = compute_layer_self_similarity(
                layer_embeddings[i],
                layer_embeddings[j],
            )

    return similarity_matrix
