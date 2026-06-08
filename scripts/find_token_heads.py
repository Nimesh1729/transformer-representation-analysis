"""Find attention heads that focus on selected target tokens."""

import pandas as pd

from src.extraction.attention_extractor import extract_attentions
from src.models.load_model import load_model, load_tokenizer
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import get_attention_dir
from src.utils.reproducibility import set_seed


def main() -> None:
    """Rank heads by attention received by target tokens."""
    logger = get_logger(__name__)

    args = parse_args()
    config = load_config(args.config)
    experiment_name = config["experiment"]["name"]
    output_dir = get_attention_dir(experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = config["model"]["name"]
    seed = config["system"]["seed"]

    set_seed(seed)

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)

    text = (
        "The quasar spectrum exhibits strong hydrogen emission lines "
        "and a high redshift."
    )

    target_tokens = {
        "qu",
        "##asa",
        "##r",
        "spectrum",
        "hydrogen",
        "emission",
        "lines",
        "red",
        "##shi",
        "##ft",
    }

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0],
    )

    attentions = extract_attentions(
        text=text,
        tokenizer=tokenizer,
        model=model,
    )

    rows = []

    for layer_index, layer_attention in enumerate(attentions):
        attention = layer_attention[0]
        num_heads = attention.shape[0]

        for head_index in range(num_heads):
            head_attention = attention[head_index]

            attention_received = head_attention.mean(dim=0)

            for token_index, token in enumerate(tokens):
                if token not in target_tokens:
                    continue

                rows.append(
                    {
                        "layer": layer_index,
                        "head": head_index,
                        "token": token,
                        "token_index": token_index,
                        "average_attention_received": float(
                            attention_received[token_index].item()
                        ),
                    }
                )

    results = pd.DataFrame(rows).sort_values(
        by="average_attention_received",
        ascending=False,
    )

    output_path = output_dir / "target_token_attention_heads.csv"
    results.to_csv(output_path, index=False)

    logger.info("Tokens: %s", tokens)
    logger.info("Saved target-token attention results to %s", output_path)
    logger.info("\n%s", results.head(20))


if __name__ == "__main__":
    main()
