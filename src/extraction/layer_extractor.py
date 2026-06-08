"""Utilities for extracting layer-wise transformer embeddings."""

import numpy as np
import torch

from src.extraction.hidden_states import extract_mean_embedding


def extract_layer_mean_embeddings(
    texts: list[str],
    tokenizer,
    model,
    layer_index: int,
) -> np.ndarray:
    """Extract mean-pooled embeddings from a specific transformer layer.

    Args:
        texts: Input text samples.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face model.
        layer_index: Index of the hidden-state layer to extract.

    Returns:
        Mean-pooled embeddings with shape (num_texts, hidden_size).

    Raises:
        IndexError: If layer_index is outside the available hidden-state range.
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states

    if layer_index < 0 or layer_index >= len(hidden_states):
        raise IndexError(
            f"layer_index {layer_index} is invalid. "
            f"Available layers: 0 to {len(hidden_states) - 1}."
        )

    layer_hidden_states = (hidden_states[layer_index],)

    embeddings = extract_mean_embedding(
        hidden_states=layer_hidden_states,
        attention_mask=inputs["attention_mask"],
    )

    return embeddings.cpu().numpy()
