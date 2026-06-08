"""Centered Kernel Alignment (CKA) utilities."""

import numpy as np


def center_gram_matrix(
    gram_matrix: np.ndarray,
) -> np.ndarray:
    """Center a Gram matrix.

    Args:
        gram_matrix: Gram matrix with shape (n_samples, n_samples).

    Returns:
        Centered Gram matrix.
    """
    n_samples = gram_matrix.shape[0]

    centering_matrix = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples

    return centering_matrix @ gram_matrix @ centering_matrix


def linear_hsic(
    features_x: np.ndarray,
    features_y: np.ndarray,
) -> float:
    """Compute linear HSIC.

    Args:
        features_x: Feature matrix of shape (n_samples, dim_x).
        features_y: Feature matrix of shape (n_samples, dim_y).

    Returns:
        HSIC score.
    """
    gram_x = features_x @ features_x.T
    gram_y = features_y @ features_y.T

    centered_x = center_gram_matrix(gram_x)
    centered_y = center_gram_matrix(gram_y)

    return float(np.sum(centered_x * centered_y))


def linear_cka(
    features_x: np.ndarray,
    features_y: np.ndarray,
) -> float:
    """Compute linear CKA similarity.

    Args:
        features_x: Feature matrix of shape (n_samples, dim_x).
        features_y: Feature matrix of shape (n_samples, dim_y).

    Returns:
        CKA similarity score.
    """
    hsic_xy = linear_hsic(features_x, features_y)
    hsic_xx = linear_hsic(features_x, features_x)
    hsic_yy = linear_hsic(features_y, features_y)

    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)
