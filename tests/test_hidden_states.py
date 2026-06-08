"""Tests for hidden-state extraction utilities."""

import torch

from src.extraction.hidden_states import (
    extract_cls_embedding,
    extract_mean_embedding,
)


def test_extract_cls_embedding_shape() -> None:
    """Test CLS extraction returns expected shape."""
    hidden_states = (
        torch.randn(2, 5, 8),
        torch.randn(2, 5, 8),
    )

    cls_embedding = extract_cls_embedding(hidden_states)

    assert cls_embedding.shape == (2, 8)


def test_extract_cls_embedding_uses_first_token() -> None:
    """Test CLS extraction selects token at index zero."""
    final_layer = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
        ]
    )
    hidden_states = (final_layer,)

    cls_embedding = extract_cls_embedding(hidden_states)

    expected = torch.tensor([[1.0, 2.0]])

    assert torch.equal(cls_embedding, expected)


def test_extract_mean_embedding_ignores_padding() -> None:
    """Test mean pooling ignores padded tokens."""
    final_layer = torch.tensor(
        [
            [
                [1.0, 1.0],
                [3.0, 3.0],
                [100.0, 100.0],
            ]
        ]
    )
    hidden_states = (final_layer,)

    attention_mask = torch.tensor([[1, 1, 0]])

    mean_embedding = extract_mean_embedding(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
    )

    expected = torch.tensor([[2.0, 2.0]])

    assert torch.equal(mean_embedding, expected)
