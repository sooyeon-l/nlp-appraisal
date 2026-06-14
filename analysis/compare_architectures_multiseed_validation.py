from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    "/content/drive/MyDrive/2026-1/NLP/appraisal/nlp-appraisal"
)

RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARCHITECTURES = {
    "Flat linear": "flat_linear",
    "Flat MLP": "flat_mlp",
    "CPM parallel": "grouped_parallel",
    "CPM sequential": "grouped_sequential",
}

SEEDS = [42, 123, 456]
LOSS = "mse"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def safe_nested_get(
    data: dict,
    keys: list[str],
    default: float = np.nan,
) -> float:
    current = data

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default

        current = current[key]

    try:
        return float(current)
    except (TypeError, ValueError):
        return default


rows = []

for architecture_label, architecture_tag in ARCHITECTURES.items():
    for seed in SEEDS:
        run_name = (
            f"{architecture_tag}_ft_{LOSS}_seed{seed}"
        )

        run_dir = RUNS_ROOT / run_name
        recomputed_path = run_dir / "val_metrics_recomputed.json"
        original_path = run_dir / "best_model.metrics.json"

        metrics_path = (
            recomputed_path
            if recomputed_path.exists()
            else original_path
        )

        if not metrics_path.exists():
            print(
                f"Missing validation metrics, skipping: "
                f"{metrics_path}"
            )
            continue

        metrics = load_json(metrics_path)

        row = {
            "Architecture": architecture_label,
            "Architecture Tag": architecture_tag,
            "Seed": seed,
            "Run": run_name,
            "Best Epoch": metrics.get("epoch", np.nan),
            "Validation Objective Loss": metrics.get(
                "best_val_objective_loss",
                np.nan,
            ),
            "Validation Macro RMSE": metrics.get(
                "macro_rmse",
                np.nan,
            ),
            "Validation Macro MAE": metrics.get(
                "macro_mae",
                np.nan,
            ),
            "Validation Macro Pearson": metrics.get(
                "macro_pearson",
                np.nan,
            ),
            "Within-entry Spearman": safe_nested_get(
                metrics,
                [
                    "ranking_metrics",
                    "mean_within_entry_spearman",
                ],
            ),
            "Exact Top-1 Accuracy": safe_nested_get(
                metrics,
                [
                    "ranking_metrics",
                    "exact_top1_accuracy",
                ],
            ),
            "Tie-aware Top-1 Accuracy": safe_nested_get(
                metrics,
                [
                    "ranking_metrics",
                    "tie_aware_top1_accuracy",
                ],
            ),
            "Top-3 Overlap": safe_nested_get(
                metrics,
                [
                    "ranking_metrics",
                    "top3_overlap",
                ],
            ),
            "Top-5 Overlap": safe_nested_get(
                metrics,
                [
                    "ranking_metrics",
                    "top5_overlap",
                ],
            ),
            "High-intensity Precision": safe_nested_get(
                metrics,
                [
                    "high_intensity_metrics",
                    "micro_precision",
                ],
            ),
            "High-intensity Recall": safe_nested_get(
                metrics,
                [
                    "high_intensity_metrics",
                    "micro_recall",
                ],
            ),
            "High-intensity F1": safe_nested_get(
                metrics,
                [
                    "high_intensity_metrics",
                    "micro_f1",
                ],
            ),
        }

        for group_name in [
            "relevance",
            "implication",
            "coping",
            "normative",
        ]:
            row[f"{group_name.title()} RMSE"] = (
                safe_nested_get(
                    metrics,
                    [
                        "group_metrics",
                        group_name,
                        "mean_rmse",
                    ],
                )
            )

            row[f"{group_name.title()} Pearson"] = (
                safe_nested_get(
                    metrics,
                    [
                        "group_metrics",
                        group_name,
                        "mean_pearson",
                    ],
                )
            )

        rows.append(row)


if not rows:
    raise RuntimeError(
        "No validation metric files were found. "
        "Check RUNS_ROOT and run naming."
    )


seed_df = pd.DataFrame(rows).sort_values(
    ["Architecture", "Seed"]
)

seed_output_path = (
    OUTPUT_DIR
    / "architecture_multiseed_validation_by_seed.csv"
)

seed_df.to_csv(
    seed_output_path,
    index=False,
)


metric_columns = [
    "Validation Macro RMSE",
    "Validation Macro MAE",
    "Validation Macro Pearson",
    "Within-entry Spearman",
    "Exact Top-1 Accuracy",
    "Tie-aware Top-1 Accuracy",
    "Top-3 Overlap",
    "Top-5 Overlap",
    "High-intensity Precision",
    "High-intensity Recall",
    "High-intensity F1",
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

for architecture, group in seed_df.groupby(
    "Architecture",
    sort=False,
):
    summary_row = {
        "Architecture": architecture,
        "Number of Seeds": len(group),
        "Seeds": ", ".join(
            str(seed)
            for seed in sorted(group["Seed"].tolist())
        ),
    }

    for metric in metric_columns:
        values = pd.to_numeric(
            group[metric],
            errors="coerce",
        ).dropna()

        summary_row[f"{metric} Mean"] = (
            float(values.mean())
            if len(values) > 0
            else np.nan
        )

        summary_row[f"{metric} SD"] = (
            float(values.std(ddof=1))
            if len(values) > 1
            else np.nan
        )

    summary_rows.append(summary_row)


summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    "Validation Macro RMSE Mean",
    ascending=True,
)

summary_output_path = (
    OUTPUT_DIR
    / "architecture_multiseed_validation_summary.csv"
)

summary_df.to_csv(
    summary_output_path,
    index=False,
)


display_columns = [
    "Architecture",
    "Number of Seeds",
    "Validation Macro RMSE Mean",
    "Validation Macro RMSE SD",
    "Validation Macro MAE Mean",
    "Validation Macro MAE SD",
    "Validation Macro Pearson Mean",
    "Validation Macro Pearson SD",
    "Within-entry Spearman Mean",
    "Within-entry Spearman SD",
    "Top-3 Overlap Mean",
    "Top-3 Overlap SD",
    "High-intensity F1 Mean",
    "High-intensity F1 SD",
]

print("\nPer-seed validation results:")
print(
    seed_df[
        [
            "Architecture",
            "Seed",
            "Validation Macro RMSE",
            "Validation Macro MAE",
            "Validation Macro Pearson",
            "Within-entry Spearman",
            "Top-3 Overlap",
            "High-intensity F1",
        ]
    ].to_string(index=False)
)

print("\nMulti-seed validation summary:")
print(
    summary_df[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print(f"\nSaved per-seed table to:\n{seed_output_path}")
print(f"\nSaved summary table to:\n{summary_output_path}")