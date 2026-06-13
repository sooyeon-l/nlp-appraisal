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

SMOOTHING_WINDOW = 50


for model_label, run_name in RUNS.items():
    path = RUNS_ROOT / run_name / "batch_losses.csv"

    if not path.exists():
        print(f"Skipping missing file: {path}")
        continue

    df = pd.read_csv(path)

    df["batch"] = range(1, len(df) + 1)

    df["smoothed_loss"] = (
        df["batch_loss"]
        .rolling(
            window=SMOOTHING_WINDOW,
            min_periods=1,
        )
        .mean()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        df["batch"],
        df["batch_loss"],
        alpha=0.2,
        linewidth=0.7,
        label="Raw batch loss",
    )

    ax.plot(
        df["batch"],
        df["smoothed_loss"],
        linewidth=1.8,
        label=f"{SMOOTHING_WINDOW}-batch moving average",
    )

    ax.set_xlabel("Optimization batch")
    ax.set_ylabel("Training objective")
    ax.set_title(f"{model_label}: batch-level training loss")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{run_name}_batch_loss.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")