"""Summarize which tokens receive the most attention in each head."""

import pandas as pd

from src.extraction.attention_extractor import extract_attentions
from src.models.load_model import load_model, load_tokenizer
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import get_attention_dir
from src.utils.reproducibility import set_seed


def main() -> None:
    """Compute top attended tokens for each attention head."""
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

    text = "The quasar spectrum shows broad emission lines."

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
        # shape: (batch, heads, query_tokens, key_tokens)
        attention = layer_attention[0]

        num_heads = attention.shape[0]

        for head_index in range(num_heads):
            head_attention = attention[head_index]

            # Average over query tokens.
            # Result shape: (key_tokens,)
            attention_received = head_attention.mean(dim=0)

            top_token_index = int(attention_received.argmax().item())
            top_token = tokens[top_token_index]
            top_score = float(attention_received[top_token_index].item())

            rows.append(
                {
                    "layer": layer_index,
                    "head": head_index,
                    "top_token": top_token,
                    "top_token_index": top_token_index,
                    "average_attention_received": top_score,
                }
            )

    summary = pd.DataFrame(rows)

    output_path = output_dir / "attention_head_summary.csv"
    summary.to_csv(output_path, index=False)

    logger.info("Saved attention head summary to %s", output_path)
    logger.info("\n%s", summary)


if __name__ == "__main__":
    main()
