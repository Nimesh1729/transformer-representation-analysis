"""Utilities for loading pretrained transformer models."""

from transformers import AutoModel, AutoTokenizer


def load_tokenizer(model_name: str):
    """Load a pretrained tokenizer from Hugging Face.

    Args:
        model_name: Hugging Face model checkpoint name.

    Returns:
        Loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name: str):
    """Load a pretrained transformer model from Hugging Face.

    Args:
        model_name: Hugging Face model checkpoint name.

    Returns:
        Loaded pretrained transformer model in evaluation mode.
    """
    model = AutoModel.from_pretrained(
        model_name,
        attn_implementation="eager",
    )
    model.eval()
    return model
