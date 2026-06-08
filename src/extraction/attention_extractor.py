"""Utilities for extracting transformer attention weights."""

import torch


def extract_attentions(
    text: str,
    tokenizer,
    model,
) -> tuple[torch.Tensor, ...]:
    """Extract attention weights for a single text input.

    Args:
        text: Input text.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face model.

    Returns:
        Attention tensors, one per transformer layer.

        Each tensor has shape:
        (batch_size, num_heads, num_tokens, num_tokens).
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
        )

    return outputs.attentions
