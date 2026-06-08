"""Tests for similarity analysis."""

import numpy as np

from src.analysis.similarity import cosine_similarity_matrix


def test_similarity_matrix_shape() -> None:
    """Test similarity matrix dimensions."""
    embeddings = np.random.randn(5, 10)

    similarity = cosine_similarity_matrix(
        embeddings,
    )

    assert similarity.shape == (5, 5)


def test_similarity_matrix_diagonal_is_one() -> None:
    """Test self-similarity."""
    embeddings = np.random.randn(5, 10)

    similarity = cosine_similarity_matrix(
        embeddings,
    )

    np.testing.assert_allclose(
        np.diag(similarity),
        np.ones(5),
        atol=1e-6,
    )
