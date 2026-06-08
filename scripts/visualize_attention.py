"""Visualize selected DistilBERT attention heads."""

from src.extraction.attention_extractor import extract_attentions
from src.models.load_model import load_model, load_tokenizer
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import get_attention_dir
from src.utils.reproducibility import set_seed
from src.visualization.attention import plot_attention_heatmap


def main() -> None:
    """Generate attention heatmaps for selected layers and heads."""
    logger = get_logger(__name__)

    args = parse_args()
    config = load_config(args.config)

    experiment_name = config["experiment"]["name"]
    output_dir = get_attention_dir(experiment_name) / "heatmaps"
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

    selected_pairs = [
        (0, 0),  # spectrum
        (0, 1),  # hydrogen
        (0, 5),  # emission
        (2, 0),  # quasar piece
        (4, 7),  # quasar piece
    ]

    for layer_index, head_index in selected_pairs:
        attention_matrix = attentions[layer_index][0, head_index].detach().cpu().numpy()

        output_path = output_dir / f"layer_{layer_index}_head_{head_index}.png"

        plot_attention_heatmap(
            attention_matrix=attention_matrix,
            tokens=tokens,
            title=f"Layer {layer_index}, Head {head_index}",
            output_path=output_path,
        )

        logger.info(
            "Saved attention heatmap for layer %d, head %d to %s",
            layer_index,
            head_index,
            output_path,
        )


if __name__ == "__main__":
    main()
