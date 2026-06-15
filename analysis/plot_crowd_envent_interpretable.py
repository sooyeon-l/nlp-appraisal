#!/usr/bin/env python
"""
Create interpretable crowd-enVENT final-test figures without duplicating the
existing per-dimension high-intensity F1 bar chart.

Outputs
-------
tables/
    crowd_envent_dimension_display_table.csv
figures/
    crowd_envent_dimension_pearson.png
    crowd_envent_gold_vs_prediction_means.png
    crowd_envent_prediction_compression.png
    crowd_envent_high_intensity_precision_recall.png

The figures answer distinct questions:
1. Which appraisal dimensions are most predictable? (Pearson)
2. Are predictions systematically shifted? (gold vs prediction means)
3. Are predictions compressed toward the middle? (gold SD vs prediction SD)
4. Why can high-intensity F1 be limited? (precision-recall tradeoff)

The script intentionally does NOT recreate the existing high-intensity F1 bar chart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_ORDER = ["relevance", "implication", "coping", "normative"]


def readable_label(name: str) -> str:
    replacements = {
        "responsblt": "responsibility",
        "conseq": "consequences",
    }
    text = str(name)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("_", " ").title()


def validate_columns(df: pd.DataFrame, required: set[str], filename: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")


def save_pearson_plot(summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary.sort_values("Pearson_Mean", ascending=True).copy()
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(9, max(7, 0.38 * len(plot_df))))
    ax.errorbar(
        plot_df["Pearson_Mean"],
        y,
        xerr=plot_df["Pearson_SD"],
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Display_Dimension"])
    ax.set_xlabel("Pearson correlation on held-out test set")
    ax.set_title("Predictability of individual appraisal dimensions")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_mean_dumbbell(bias: pd.DataFrame, output_path: Path) -> None:
    plot_df = bias.copy()
    plot_df["abs_bias"] = plot_df["Mean_Bias"].abs()
    plot_df = plot_df.sort_values("abs_bias", ascending=True)
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(9.5, max(7, 0.38 * len(plot_df))))
    for i, row in enumerate(plot_df.itertuples(index=False)):
        ax.plot([row.Gold_Mean, row.Prediction_Mean], [i, i], linewidth=1.5)
    ax.scatter(plot_df["Gold_Mean"], y, label="Gold mean")
    ax.scatter(plot_df["Prediction_Mean"], y, marker="s", label="Prediction mean")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Display_Dimension"])
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Mean normalized appraisal score")
    ax.set_title("Gold versus predicted mean by appraisal dimension")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_compression_plot(bias: pd.DataFrame, output_path: Path) -> None:
    plot_df = bias.sort_values("SD_Ratio", ascending=True).copy()
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(9, max(7, 0.38 * len(plot_df))))
    ax.scatter(plot_df["SD_Ratio"], y)
    ax.axvline(1.0, linestyle="--", linewidth=1, label="Equal variability")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Display_Dimension"])
    ax.set_xlabel("Prediction SD / gold SD")
    ax.set_title("Prediction-range compression by appraisal dimension")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_plot(summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary.copy()

    fig, ax = plt.subplots(figsize=(8.5, 7))
    for group, group_df in plot_df.groupby("Group", sort=False):
        ax.scatter(
            group_df["High_Recall_Mean"],
            group_df["High_Precision_Mean"],
            label=str(group).title(),
            s=45,
        )

    for _, row in plot_df.iterrows():
        ax.annotate(
            row["Display_Dimension"],
            (row["High_Recall_Mean"], row["High_Precision_Mean"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7.5,
        )

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("High-intensity recall")
    ax.set_ylabel("High-intensity precision")
    ax.set_title("High-intensity detection: precision–recall tradeoff")
    ax.legend(frameon=False, title="CPM stage")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension-summary", type=Path, required=True)
    parser.add_argument("--dimension-by-seed", type=Path, required=True)
    parser.add_argument("--bias-summary", type=Path, required=True)
    parser.add_argument("--bias-by-seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.dimension_summary)
    by_seed = pd.read_csv(args.dimension_by_seed)
    bias = pd.read_csv(args.bias_summary)
    bias_by_seed = pd.read_csv(args.bias_by_seed)

    validate_columns(
        summary,
        {
            "Dimension",
            "Group",
            "Pearson_Mean",
            "Pearson_SD",
            "High_Precision_Mean",
            "High_Recall_Mean",
            "High_F1_Mean",
        },
        str(args.dimension_summary),
    )
    validate_columns(
        bias,
        {
            "Dimension",
            "Gold_Mean",
            "Prediction_Mean",
            "Mean_Bias",
            "Gold_SD",
            "Prediction_SD",
            "SD_Ratio",
            "Gold_High_Rate",
            "Predicted_High_Rate",
        },
        str(args.bias_summary),
    )

    # Ensure the by-seed files are compatible and genuinely contain three-seed results.
    validate_columns(
        by_seed,
        {"Seed", "Dimension", "Pearson", "High-intensity F1"},
        str(args.dimension_by_seed),
    )
    validate_columns(
        bias_by_seed,
        {"Seed", "Dimension", "Mean Bias", "SD Ratio"},
        str(args.bias_by_seed),
    )

    n_seeds_metrics = by_seed.groupby("Dimension")["Seed"].nunique()
    n_seeds_bias = bias_by_seed.groupby("Dimension")["Seed"].nunique()
    if (n_seeds_metrics < 2).any() or (n_seeds_bias < 2).any():
        print("Warning: at least one dimension contains fewer than two seeds.")

    summary["Display_Dimension"] = summary["Dimension"].map(readable_label)
    bias["Display_Dimension"] = bias["Dimension"].map(readable_label)

    display = summary.merge(
        bias[
            [
                "Dimension",
                "Gold_Mean",
                "Prediction_Mean",
                "Mean_Bias",
                "Gold_SD",
                "Prediction_SD",
                "SD_Ratio",
                "Gold_High_Rate",
                "Predicted_High_Rate",
                "High_Rate_Difference",
            ]
        ],
        on="Dimension",
        how="left",
        validate="one_to_one",
    )
    display.to_csv(
        table_dir / "crowd_envent_dimension_display_table.csv",
        index=False,
    )

    save_pearson_plot(
        summary,
        figure_dir / "crowd_envent_dimension_pearson.png",
    )
    save_mean_dumbbell(
        bias,
        figure_dir / "crowd_envent_gold_vs_prediction_means.png",
    )
    save_compression_plot(
        bias,
        figure_dir / "crowd_envent_prediction_compression.png",
    )
    save_precision_recall_plot(
        summary,
        figure_dir / "crowd_envent_high_intensity_precision_recall.png",
    )

    print("\nHighest Pearson dimensions:")
    print(
        summary.nlargest(5, "Pearson_Mean")[
            ["Display_Dimension", "Group", "Pearson_Mean", "Pearson_SD"]
        ].round(4).to_string(index=False)
    )
    print("\nMost compressed prediction ranges:")
    print(
        bias.nsmallest(5, "SD_Ratio")[
            ["Display_Dimension", "Gold_SD", "Prediction_SD", "SD_Ratio"]
        ].round(4).to_string(index=False)
    )
    print("\nLargest absolute mean biases:")
    temp = bias.assign(abs_bias=bias["Mean_Bias"].abs())
    print(
        temp.nlargest(5, "abs_bias")[
            ["Display_Dimension", "Gold_Mean", "Prediction_Mean", "Mean_Bias"]
        ].round(4).to_string(index=False)
    )
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
