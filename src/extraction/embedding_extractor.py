"""Utilities for extracting transformer embeddings from text."""

import numpy as np
import torch

from src.extraction.hidden_states import (
    extract_cls_embedding,
    extract_mean_embedding,
)


def extract_cls_embeddings(
    texts: list[str],
    tokenizer,
    model,
) -> np.ndarray:
    """Extract CLS embeddings from input texts.

    Args:
        texts: Input text samples.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face model.

    Returns:
        CLS embeddings with shape (num_texts, hidden_size).
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

    cls_embeddings = extract_cls_embedding(outputs.hidden_states)

    return cls_embeddings.cpu().numpy()


def extract_mean_embeddings(
    texts: list[str],
    tokenizer,
    model,
) -> np.ndarray:
    """Extract mean-pooled embeddings from input texts.

    Args:
        texts: Input text samples.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face model.

    Returns:
        Mean-pooled embeddings with shape (num_texts, hidden_size).
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

    mean_embeddings = extract_mean_embedding(
        hidden_states=outputs.hidden_states,
        attention_mask=inputs["attention_mask"],
    )

    return mean_embeddings.cpu().numpy()
