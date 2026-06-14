from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.config import (
    SAVE_PATH,
    OBJECTIVE_GROUPS,
    TARGET_DIMS,
)


# ============================================================
# Paths
# ============================================================

DATA_ROOT = Path(SAVE_PATH)
RUNS_ROOT = DATA_ROOT / "runs"
OUTPUT_DIR = DATA_ROOT / "outputs" / "eval"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Final selected runs
# ============================================================

FINAL_TEST_RUNS = {
    42: "grouped_sequential_ft_mse_seed42",
    123: "grouped_sequential_ft_mse_seed123",
    456: "grouped_sequential_ft_mse_seed456",
}

HIGH_INTENSITY_THRESHOLD = 0.75


# ============================================================
# Helpers
# ============================================================

def dimension_to_group(
    dimension: str,
) -> str:
    for group_name, group_dims in OBJECTIVE_GROUPS.items():
        if dimension in group_dims:
            return group_name

    return "unknown"


def safe_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    if len(y_true) < 2:
        return np.nan

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    correlation = np.corrcoef(
        y_true,
        y_pred,
    )[0, 1]

    return (
        float(correlation)
        if np.isfinite(correlation)
        else np.nan
    )


def compute_dimension_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {
            "N": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Pearson": np.nan,
            "High-intensity Precision": np.nan,
            "High-intensity Recall": np.nan,
            "High-intensity F1": np.nan,
            "High-intensity Support": 0,
        }

    errors = y_pred - y_true

    rmse = float(
        np.sqrt(np.mean(errors ** 2))
    )

    mae = float(
        np.mean(np.abs(errors))
    )

    pearson = safe_pearson(
        y_true=y_true,
        y_pred=y_pred,
    )

    gold_binary = (
        y_true >= HIGH_INTENSITY_THRESHOLD
    ).astype(int)

    pred_binary = (
        y_pred >= HIGH_INTENSITY_THRESHOLD
    ).astype(int)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            gold_binary,
            pred_binary,
            average="binary",
            zero_division=0,
        )
    )

    return {
        "N": len(y_true),
        "RMSE": rmse,
        "MAE": mae,
        "Pearson": pearson,
        "High-intensity Precision": float(precision),
        "High-intensity Recall": float(recall),
        "High-intensity F1": float(f1),
        "High-intensity Support": int(
            np.sum(gold_binary)
        ),
    }


# ============================================================
# Load prediction CSVs and calculate per-dimension metrics
# ============================================================

rows = []

reference_true = None
reference_valid_mask = None

for seed, run_name in FINAL_TEST_RUNS.items():
    path = (
        RUNS_ROOT
        / run_name
        / "test_predictions_recomputed.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing test prediction file:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        f"{dimension}_{suffix}"
        for dimension in TARGET_DIMS
        for suffix in ("true", "pred")
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {path}:\n"
            + "\n".join(missing_columns)
        )

    true_matrix = np.column_stack([
        df[f"{dimension}_true"].to_numpy(dtype=float)
        for dimension in TARGET_DIMS
    ])

    pred_matrix = np.column_stack([
        df[f"{dimension}_pred"].to_numpy(dtype=float)
        for dimension in TARGET_DIMS
    ])

    valid_mask = np.isfinite(true_matrix)

    if reference_true is None:
        reference_true = true_matrix
        reference_valid_mask = valid_mask
    else:
        if not np.array_equal(
            valid_mask,
            reference_valid_mask,
        ):
            raise ValueError(
                f"Gold-label mask mismatch for seed {seed}"
            )

        if not np.allclose(
            true_matrix[valid_mask],
            reference_true[valid_mask],
            atol=1e-8,
        ):
            raise ValueError(
                f"Gold-label mismatch for seed {seed}"
            )

    for dim_idx, dimension in enumerate(TARGET_DIMS):
        metrics = compute_dimension_metrics(
            y_true=true_matrix[:, dim_idx],
            y_pred=pred_matrix[:, dim_idx],
        )

        rows.append({
            "Seed": seed,
            "Dimension": dimension,
            "Group": dimension_to_group(
                dimension
            ),
            **metrics,
        })


# ============================================================
# Seed-level and three-seed summaries
# ============================================================

by_seed_df = pd.DataFrame(rows)

summary_df = (
    by_seed_df
    .groupby(
        ["Dimension", "Group"],
        as_index=False,
    )
    .agg(
        N=("N", "max"),
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
        High_Precision_SD=(
            "High-intensity Precision",
            "std",
        ),
        High_Recall_Mean=(
            "High-intensity Recall",
            "mean",
        ),
        High_Recall_SD=(
            "High-intensity Recall",
            "std",
        ),
        High_F1_Mean=(
            "High-intensity F1",
            "mean",
        ),
        High_F1_SD=(
            "High-intensity F1",
            "std",
        ),
        High_Support=(
            "High-intensity Support",
            "max",
        ),
    )
)


# Normalized labels are on a 0–1 scale.
# Multiplying error metrics by 4 converts them to the
# original 1–5 appraisal scale.
summary_df["Original_Scale_RMSE_Mean"] = (
    summary_df["RMSE_Mean"] * 4
)

summary_df["Original_Scale_RMSE_SD"] = (
    summary_df["RMSE_SD"] * 4
)

summary_df["Original_Scale_MAE_Mean"] = (
    summary_df["MAE_Mean"] * 4
)

summary_df["Original_Scale_MAE_SD"] = (
    summary_df["MAE_SD"] * 4
)


# Preserve the target-dimension order from config.py.
dimension_order = {
    dimension: index
    for index, dimension in enumerate(TARGET_DIMS)
}

by_seed_df["_order"] = (
    by_seed_df["Dimension"].map(dimension_order)
)

summary_df["_order"] = (
    summary_df["Dimension"].map(dimension_order)
)

by_seed_df = (
    by_seed_df
    .sort_values(["_order", "Seed"])
    .drop(columns="_order")
    .reset_index(drop=True)
)

summary_df = (
    summary_df
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)


# ============================================================
# Save
# ============================================================

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


# ============================================================
# Console summaries
# ============================================================

display_columns = [
    "Dimension",
    "Group",
    "RMSE_Mean",
    "Pearson_Mean",
    "High_F1_Mean",
    "High_Support",
]


print("\nBest dimensions by RMSE:")
print(
    summary_df
    .sort_values("RMSE_Mean")
    [display_columns]
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
    [display_columns]
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
    [display_columns]
    .head(10)
    .to_string(index=False)
)


print("\nWorst dimensions by Pearson:")
print(
    summary_df
    .sort_values("Pearson_Mean")
    [display_columns]
    .head(10)
    .to_string(index=False)
)


print("\nSaved:")
print(by_seed_path)
print(summary_path)
