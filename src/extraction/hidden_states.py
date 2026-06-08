"""Utilities for extracting embeddings from transformer hidden states."""

import torch


def extract_cls_embedding(
    hidden_states: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Extract final-layer CLS embedding.

    Args:
        hidden_states: Hidden states returned by the transformer.

    Returns:
        CLS embedding with shape (batch_size, hidden_size).
    """
    return hidden_states[-1][:, 0, :]


def extract_mean_embedding(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract final-layer mean-pooled embedding.

    Padding tokens are ignored using the attention mask.

    Args:
        hidden_states: Hidden states returned by the transformer.
        attention_mask: Attention mask with shape (batch_size, num_tokens).

    Returns:
        Mean-pooled embedding with shape (batch_size, hidden_size).
    """
    last_hidden_state = hidden_states[-1]

    mask = attention_mask.unsqueeze(-1)
    masked_hidden_state = last_hidden_state * mask

    sum_embeddings = masked_hidden_state.sum(dim=1)
    token_counts = mask.sum(dim=1)

    return sum_embeddings / token_counts
