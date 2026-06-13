from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUNS_ROOT = Path("/workspace/data/runs")
OUTPUT_DIR = Path("/workspace/data/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "Flat linear": "flat_linear_ft_weighted",
    "Flat MLP": "flat_mlp_ft_weighted",
    "CPM parallel": "grouped_parallel_ft_weighted",
    "CPM sequential": "grouped_sequential_ft_weighted",
}


for model_label, run_name in RUNS.items():
    log_path = RUNS_ROOT / run_name / "training_log.csv"

    if not log_path.exists():
        print(f"Skipping missing log: {log_path}")
        continue

    df = pd.read_csv(log_path)

    fig, ax = plt.subplots(figsize=(6.5, 4))

    ax.plot(
        df["epoch"],
        df["train_objective_loss"],
        marker="o",
        label="Training",
    )

    ax.plot(
        df["epoch"],
        df["val_objective_loss"],
        marker="o",
        label="Validation",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Objective loss")
    ax.set_title(f"{model_label}: training and validation loss")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{run_name}_epoch_training_curve.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")