from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RUNS_ROOT = Path("/workspace/data/runs")
OUTPUT_DIR = Path("/workspace/data/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RUNS = {
    "MSE": "grouped_sequential_ft_mse_seed42",
    "Weighted MSE": "grouped_sequential_ft_weighted",
    "Group-balanced MSE": "grouped_sequential_ft_group_balanced_seed42",
    "Weighted group-balanced MSE":
        "grouped_sequential_ft_weighted_group_balanced_seed42",
}


rows = []

for loss_label, run_name in RUNS.items():
    metrics_path = RUNS_ROOT / run_name / "best_model.metrics.json"

    if not metrics_path.exists():
        print(f"Missing: {metrics_path}")
        continue

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    row = {
        "Loss": loss_label,
        "Run": run_name,
        "Validation Macro RMSE": metrics["macro_rmse"],
        "Validation Macro MAE": metrics["macro_mae"],
        "Validation Macro Pearson": metrics["macro_pearson"],
    }

    for group_name, group_metrics in metrics["group_metrics"].items():
        row[f"{group_name.title()} RMSE"] = (
            group_metrics["mean_rmse"]
        )
        row[f"{group_name.title()} Pearson"] = (
            group_metrics["mean_pearson"]
        )

    rows.append(row)


df = pd.DataFrame(rows)

df = df.sort_values(
    "Validation Macro RMSE",
    ascending=True,
).reset_index(drop=True)

output_path = OUTPUT_DIR / "loss_ablation_validation.csv"
df.to_csv(output_path, index=False)

print(df.round(4).to_string(index=False))
print(f"\nSaved: {output_path}")