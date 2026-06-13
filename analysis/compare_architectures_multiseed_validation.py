from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RUNS_ROOT = Path("/workspace/data/runs")
OUTPUT_DIR = Path("/workspace/data/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOSS_TAG = "weighted"

MODELS = {
    "Flat linear": "flat_linear",
    "Flat MLP": "flat_mlp",
    "CPM parallel": "grouped_parallel",
    "CPM sequential": "grouped_sequential",
}

SEEDS = [42, 123, 456]


def get_run_name(model_name: str, seed: int) -> str:
    if seed == 42:
        return f"{model_name}_ft_{LOSS_TAG}"

    return f"{model_name}_ft_{LOSS_TAG}_seed{seed}"


rows = []

for model_label, model_name in MODELS.items():
    for seed in SEEDS:
        run_name = get_run_name(model_name, seed)

        metrics_path = (
            RUNS_ROOT
            / run_name
            / "best_model.metrics.json"
        )

        if not metrics_path.exists():
            print(f"Missing: {metrics_path}")
            continue

        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        row = {
            "Model": model_label,
            "Run": run_name,
            "Seed": seed,
            "Macro RMSE": metrics["macro_rmse"],
            "Macro MAE": metrics["macro_mae"],
            "Macro Pearson": metrics["macro_pearson"],
        }

        for group_name, group_metrics in metrics[
            "group_metrics"
        ].items():
            row[f"{group_name.title()} RMSE"] = (
                group_metrics["mean_rmse"]
            )
            row[f"{group_name.title()} Pearson"] = (
                group_metrics["mean_pearson"]
            )

        rows.append(row)


long_df = pd.DataFrame(rows)

long_df.to_csv(
    OUTPUT_DIR / "architecture_multiseed_validation_long.csv",
    index=False,
)


metric_columns = [
    "Macro RMSE",
    "Macro MAE",
    "Macro Pearson",
    "Relevance RMSE",
    "Relevance Pearson",
    "Implication RMSE",
    "Implication Pearson",
    "Coping RMSE",
    "Coping Pearson",
    "Normative RMSE",
    "Normative Pearson",
]


summary_rows = []

for model_label, model_df in long_df.groupby("Model"):
    row = {
        "Model": model_label,
        "N Seeds": len(model_df),
    }

    for metric in metric_columns:
        row[f"{metric} Mean"] = model_df[metric].mean()
        row[f"{metric} SD"] = model_df[metric].std(ddof=1)

    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    "Macro RMSE Mean",
    ascending=True,
).reset_index(drop=True)

summary_df.to_csv(
    OUTPUT_DIR / "architecture_multiseed_validation_summary.csv",
    index=False,
)

print(summary_df.round(4).to_string(index=False))