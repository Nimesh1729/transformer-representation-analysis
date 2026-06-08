"""Inspect DistilBERT attention shapes for a sample sentence."""

from src.extraction.attention_extractor import extract_attentions
from src.models.load_model import load_model, load_tokenizer
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed


def main() -> None:
    """Extract and inspect attention weights."""
    logger = get_logger(__name__)

    args = parse_args()
    config = load_config(args.config)
    model_name = config["model"]["name"]
    seed = config["system"]["seed"]

    set_seed(seed)

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)

    text = "The quasar spectrum shows broad emission lines."

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    decoded_tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0],
    )

    decoded_sentence = tokenizer.decode(
        inputs["input_ids"][0],
    )

    logger.info("Original text: %s", text)
    logger.info("Decoded tokens: %s", decoded_tokens)
    logger.info("Decoded sentence: %s", decoded_sentence)

    attentions = extract_attentions(
        text=text,
        tokenizer=tokenizer,
        model=model,
    )

    logger.info("Tokens: %s", decoded_sentence)
    logger.info("Number of attention tensors: %d", len(attentions))

    for layer_index, attention in enumerate(attentions):
        logger.info(
            "Layer %d attention shape: %s",
            layer_index,
            tuple(attention.shape),
        )


if __name__ == "__main__":
    main()
