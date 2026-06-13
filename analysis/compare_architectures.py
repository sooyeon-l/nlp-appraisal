from __future__ import annotations

import json
from pathlib import Path

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


rows = []

for model_label, run_name in RUNS.items():
    run_dir = RUNS_ROOT / run_name

    with open(
        run_dir / "test_metrics.json",
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    group_metrics = metrics["group_metrics"]

    row = {
        "Model": model_label,
        "Run": run_name,
        "Macro RMSE": metrics["macro_rmse"],
        "Macro MAE": metrics["macro_mae"],
        "Macro Pearson": metrics["macro_pearson"],
    }

    for group_name in [
        "relevance",
        "implication",
        "coping",
        "normative",
    ]:
        row[f"{group_name.title()} RMSE"] = (
            group_metrics[group_name]["mean_rmse"]
        )

        row[f"{group_name.title()} Pearson"] = (
            group_metrics[group_name]["mean_pearson"]
        )

    rows.append(row)


comparison_df = pd.DataFrame(rows)
comparison_df["Macro RMSE (1-5 scale)"] = (
    comparison_df["Macro RMSE"] * 4
)

comparison_df["Difference from paper"] = (
    comparison_df["Macro RMSE (1-5 scale)"] - 1.40
)

comparison_df.to_csv(
    OUTPUT_DIR / "architecture_comparison.csv",
    index=False,
)

print(comparison_df.round(4).to_string(index=False))