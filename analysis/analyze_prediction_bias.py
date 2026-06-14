from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SAVE_PATH, TARGET_DIMS

DATA_ROOT = Path(SAVE_PATH)
RUNS_ROOT = DATA_ROOT / "runs"
OUTPUT_DIR = DATA_ROOT / "outputs" / "eval"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FINAL_TEST_RUNS = {
    42: "grouped_sequential_ft_mse_seed42",
    123: "grouped_sequential_ft_mse_seed123",
    456: "grouped_sequential_ft_mse_seed456",
}


rows = []

for seed, run_name in FINAL_TEST_RUNS.items():
    path = (
        RUNS_ROOT
        / run_name
        / "test_predictions_recomputed.csv"
    )

    df = pd.read_csv(path)

    for dimension in TARGET_DIMS:
        gold = df[
            f"{dimension}_true"
        ].to_numpy(dtype=float)

        pred = df[
            f"{dimension}_pred"
        ].to_numpy(dtype=float)

        valid = np.isfinite(gold)

        gold = gold[valid]
        pred = pred[valid]

        rows.append({
            "Seed": seed,
            "Dimension": dimension,
            "Gold Mean": np.mean(gold),
            "Prediction Mean": np.mean(pred),
            "Mean Bias": np.mean(pred - gold),
            "Gold SD": np.std(
                gold,
                ddof=1,
            ),
            "Prediction SD": np.std(
                pred,
                ddof=1,
            ),
            "SD Ratio": (
                np.std(pred, ddof=1)
                / np.std(gold, ddof=1)
                if np.std(gold, ddof=1) > 0
                else np.nan
            ),
            "Gold High Rate": np.mean(
                gold >= 0.75
            ),
            "Predicted High Rate": np.mean(
                pred >= 0.75
            ),
        })


bias_seed_df = pd.DataFrame(rows)

bias_summary_df = (
    bias_seed_df
    .groupby(
        "Dimension",
        as_index=False,
    )
    .agg(
        Gold_Mean=("Gold Mean", "mean"),
        Prediction_Mean=(
            "Prediction Mean",
            "mean",
        ),
        Mean_Bias=("Mean Bias", "mean"),
        Gold_SD=("Gold SD", "mean"),
        Prediction_SD=(
            "Prediction SD",
            "mean",
        ),
        SD_Ratio=("SD Ratio", "mean"),
        Gold_High_Rate=(
            "Gold High Rate",
            "mean",
        ),
        Predicted_High_Rate=(
            "Predicted High Rate",
            "mean",
        ),
    )
)

bias_summary_df["High_Rate_Difference"] = (
    bias_summary_df["Predicted_High_Rate"]
    - bias_summary_df["Gold_High_Rate"]
)


bias_seed_path = (
    OUTPUT_DIR
    / "final_test_prediction_bias_by_seed.csv"
)

bias_summary_path = (
    OUTPUT_DIR
    / "final_test_prediction_bias_summary.csv"
)

bias_seed_df.to_csv(
    bias_seed_path,
    index=False,
)

bias_summary_df.to_csv(
    bias_summary_path,
    index=False,
)


print("\nMost underpredicted dimensions:")
print(
    bias_summary_df
    .sort_values("Mean_Bias")
    .head(10)
    .to_string(index=False)
)


print("\nMost overpredicted dimensions:")
print(
    bias_summary_df
    .sort_values(
        "Mean_Bias",
        ascending=False,
    )
    .head(10)
    .to_string(index=False)
)


print("\nMost compressed prediction distributions:")
print(
    bias_summary_df
    .sort_values("SD_Ratio")
    .head(10)
    [
        [
            "Dimension",
            "Gold_SD",
            "Prediction_SD",
            "SD_Ratio",
            "Gold_High_Rate",
            "Predicted_High_Rate",
        ]
    ]
    .to_string(index=False)
)


print("\nSaved:")
print(bias_seed_path)
print(bias_summary_path)