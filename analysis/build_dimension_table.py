from pathlib import Path
import json

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


all_rows = []

for model_label, run_name in RUNS.items():
    metrics_path = (
        RUNS_ROOT
        / run_name
        / "test_metrics.json"
    )

    with open(
        metrics_path,
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    dimensions = metrics["per_dim_rmse"].keys()

    for dim_name in dimensions:
        all_rows.append({
            "Model": model_label,
            "Run": run_name,
            "Dimension": dim_name,
            "RMSE": metrics["per_dim_rmse"][dim_name],
            "MAE": metrics["per_dim_mae"][dim_name],
            "Pearson": metrics["per_dim_pearson"][dim_name],
        })


long_df = pd.DataFrame(all_rows)

long_path = (
    OUTPUT_DIR
    / "per_dimension_results_long.csv"
)

long_df.to_csv(
    long_path,
    index=False,
)


rmse_wide = long_df.pivot(
    index="Dimension",
    columns="Model",
    values="RMSE",
)

pearson_wide = long_df.pivot(
    index="Dimension",
    columns="Model",
    values="Pearson",
)

rmse_wide.to_csv(
    OUTPUT_DIR / "per_dimension_rmse_table.csv"
)

pearson_wide.to_csv(
    OUTPUT_DIR / "per_dimension_pearson_table.csv"
)

print("Saved:")
print(long_path)
print(OUTPUT_DIR / "per_dimension_rmse_table.csv")
print(OUTPUT_DIR / "per_dimension_pearson_table.csv")