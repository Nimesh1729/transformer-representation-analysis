"""Run the full representation-analysis pipeline."""

import argparse
import subprocess
import sys
from datetime import datetime

from src.utils.config_loader import load_config
from src.utils.paths import get_logs_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the full transformer representation-analysis pipeline."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file.",
    )

    parser.add_argument(
        "--embedding-type",
        choices=["cls", "mean"],
        default="mean",
        help="Embedding type to use for embedding-level analyses.",
    )

    return parser.parse_args()


def run_command(command: list[str], log_file) -> None:
    """Run a command and write output to a log file.

    Args:
        command: Command arguments.
        log_file: Open log file handle.
    """
    command_text = " ".join(command)

    print(f"Running: {command_text}")
    log_file.write(f"\n\nRunning: {command_text}\n")
    log_file.flush()

    subprocess.run(
        command,
        check=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )


def build_commands(config_path: str, embedding_type: str) -> list[list[str]]:
    """Build full pipeline commands.

    Args:
        config_path: Path to experiment config.
        embedding_type: Embedding type, either cls or mean.

    Returns:
        List of commands to run.
    """
    return [
        [sys.executable, "main.py", "--config", config_path],
        [
            sys.executable,
            "scripts/run_visual_analysis.py",
            "--config",
            config_path,
            "--embedding-type",
            embedding_type,
        ],
        [
            sys.executable,
            "scripts/run_similarity_analysis.py",
            "--config",
            config_path,
            "--embedding-type",
            embedding_type,
        ],
        [
            sys.executable,
            "scripts/extract_layer_embeddings.py",
            "--config",
            config_path,
        ],
        [sys.executable, "scripts/run_layer_similarity.py", "--config", config_path],
        [sys.executable, "scripts/run_layer_cka.py", "--config", config_path],
        [
            sys.executable,
            "scripts/run_layer_class_similarity.py",
            "--config",
            config_path,
        ],
        [sys.executable, "scripts/analyze_best_layer.py", "--config", config_path],
        [
            sys.executable,
            "scripts/run_layer_visualizations.py",
            "--config",
            config_path,
        ],
    ]


def main() -> None:
    """Run the full analysis pipeline."""
    args = parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment"]["name"]

    logs_dir = get_logs_dir(experiment_name)
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = logs_dir / f"{timestamp}_{args.embedding_type}.log"

    commands = build_commands(
        config_path=args.config,
        embedding_type=args.embedding_type,
    )

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Experiment: {experiment_name}\n")
        log_file.write(f"Config: {args.config}\n")
        log_file.write(f"Embedding type: {args.embedding_type}\n")
        log_file.write(f"Python executable: {sys.executable}\n")

        for command in commands:
            run_command(command, log_file)

    print(f"Pipeline complete. Log saved to: {log_path}")


if __name__ == "__main__":
    main()
