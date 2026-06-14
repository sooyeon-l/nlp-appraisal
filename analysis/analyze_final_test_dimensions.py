from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    OBJECTIVE_GROUPS,
    TARGET_DIMS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FINAL_TEST_RUNS = {
    42: "grouped_sequential_ft_mse_seed42",
    123: "grouped_sequential_ft_mse_seed123",
    456: "grouped_sequential_ft_mse_seed456",
}


def dimension_to_group(
    dimension: str,
) -> str:
    for group_name, group_dims in (
        OBJECTIVE_GROUPS.items()
    ):
        if dimension in group_dims:
            return group_name

    return "unknown"


rows = []

for seed, run_name in FINAL_TEST_RUNS.items():
    path = (
        RUNS_ROOT
        / run_name
        / "test_metrics_recomputed.json"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    high_metrics = metrics[
        "high_intensity_metrics"
    ].get("per_dimension", {})

    for dimension in TARGET_DIMS:
        dimension_high = high_metrics.get(
            dimension,
            {},
        )

        rows.append({
            "Seed": seed,
            "Dimension": dimension,
            "Group": dimension_to_group(
                dimension
            ),
            "RMSE": metrics[
                "per_dim_rmse"
            ][dimension],
            "MAE": metrics[
                "per_dim_mae"
            ][dimension],
            "Pearson": metrics[
                "per_dim_pearson"
            ][dimension],
            "High-intensity Precision":
                dimension_high.get(
                    "precision",
                    np.nan,
                ),
            "High-intensity Recall":
                dimension_high.get(
                    "recall",
                    np.nan,
                ),
            "High-intensity F1":
                dimension_high.get(
                    "f1",
                    np.nan,
                ),
            "High-intensity Support":
                dimension_high.get(
                    "support",
                    np.nan,
                ),
        })


by_seed_df = pd.DataFrame(rows)

summary_df = (
    by_seed_df
    .groupby(
        ["Dimension", "Group"],
        as_index=False,
    )
    .agg(
        RMSE_Mean=("RMSE", "mean"),
        RMSE_SD=("RMSE", "std"),
        MAE_Mean=("MAE", "mean"),
        MAE_SD=("MAE", "std"),
        Pearson_Mean=("Pearson", "mean"),
        Pearson_SD=("Pearson", "std"),
        High_Precision_Mean=(
            "High-intensity Precision",
            "mean",
        ),
        High_Recall_Mean=(
            "High-intensity Recall",
            "mean",
        ),
        High_F1_Mean=(
            "High-intensity F1",
            "mean",
        ),
        High_F1_SD=(
            "High-intensity F1",
            "std",
        ),
        High_Support_Mean=(
            "High-intensity Support",
            "mean",
        ),
    )
)


summary_df["Original_Scale_RMSE_Mean"] = (
    summary_df["RMSE_Mean"] * 4
)

summary_df["Original_Scale_RMSE_SD"] = (
    summary_df["RMSE_SD"] * 4
)


by_seed_path = (
    OUTPUT_DIR
    / "final_test_dimensions_by_seed.csv"
)

summary_path = (
    OUTPUT_DIR
    / "final_test_dimensions_summary.csv"
)

by_seed_df.to_csv(
    by_seed_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)


print("\nBest dimensions by RMSE:")
print(
    summary_df
    .sort_values("RMSE_Mean")
    [
        [
            "Dimension",
            "Group",
            "RMSE_Mean",
            "Pearson_Mean",
            "High_F1_Mean",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print("\nHardest dimensions by RMSE:")
print(
    summary_df
    .sort_values(
        "RMSE_Mean",
        ascending=False,
    )
    [
        [
            "Dimension",
            "Group",
            "RMSE_Mean",
            "Pearson_Mean",
            "High_F1_Mean",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print("\nBest dimensions by Pearson:")
print(
    summary_df
    .sort_values(
        "Pearson_Mean",
        ascending=False,
    )
    [
        [
            "Dimension",
            "Group",
            "RMSE_Mean",
            "Pearson_Mean",
            "High_F1_Mean",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print("\nWorst dimensions by Pearson:")
print(
    summary_df
    .sort_values("Pearson_Mean")
    [
        [
            "Dimension",
            "Group",
            "RMSE_Mean",
            "Pearson_Mean",
            "High_F1_Mean",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print("\nSaved:")
print(by_seed_path)
print(summary_path)