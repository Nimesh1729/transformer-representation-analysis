"""Tests for reproducibility utilities."""

import random

import numpy as np
import torch

from src.utils.reproducibility import set_seed


def test_set_seed_makes_random_outputs_reproducible() -> None:
    """Test that set_seed makes random generators reproducible."""
    set_seed(42)

    python_random_1 = random.random()
    numpy_random_1 = np.random.rand()
    torch_random_1 = torch.rand(1)

    set_seed(42)

    python_random_2 = random.random()
    numpy_random_2 = np.random.rand()
    torch_random_2 = torch.rand(1)

    assert python_random_1 == python_random_2
    assert numpy_random_1 == numpy_random_2
    assert torch.equal(torch_random_1, torch_random_2)
