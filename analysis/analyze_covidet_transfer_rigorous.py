#!/usr/bin/env python
"""
Rigorous zero-shot transfer analysis for crowd-enVENT -> CovidET.

Outputs
-------
tables/
    covidet_dimension_transfer.csv
    covidet_condition_macro_summary.csv
    covidet_truncation_bootstrap.csv
figures/
    covidet_dimension_correlations_<condition>.png
    covidet_gold_vs_prediction_means_<condition>.png
    covidet_selected_calibration_<condition>.png
    covidet_truncation_effects.png

The script:
1. evaluates only mappings marked include_primary_eval=True by default;
2. separates discrimination (Pearson/Spearman) from absolute agreement
   (MAE/RMSE/bias/calibration);
3. compares head, tail, head-tail, and sliding-window inference;
4. performs paired bootstrap resampling over posts for truncation comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


CONDITION_LABELS = {
    "head": "Head",
    "tail": "Tail",
    "head_tail": "Head–tail",
    "sliding": "Sliding window",
}

DEFAULT_FILES = {
    "head": "ensemble_head_128.csv",
    "tail": "ensemble_tail_128.csv",
    "head_tail": "ensemble_head_tail_128.csv",
    "sliding": "ensemble_sliding_window_128_stride64.csv",
}


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError(f"Unknown correlation method: {method}")


def calibration_stats(pred: np.ndarray, gold: np.ndarray) -> Tuple[float, float, float]:
    mask = np.isfinite(pred) & np.isfinite(gold)
    pred = pred[mask]
    gold = gold[mask]
    if len(pred) < 3 or np.nanstd(pred) == 0:
        return np.nan, np.nan, np.nan

    model = LinearRegression().fit(pred.reshape(-1, 1), gold)
    return (
        float(model.intercept_),
        float(model.coef_[0]),
        float(model.score(pred.reshape(-1, 1), gold)),
    )


def load_mapping(path: Path, primary_only: bool = True) -> pd.DataFrame:
    mapping = pd.read_csv(path)
    required = {
        "crowd_envent_dimension",
        "covidet_column",
        "covidet_dimension",
        "direction",
        "include_primary_eval",
    }
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping file is missing columns: {sorted(missing)}")

    mapping = mapping[mapping["covidet_column"].notna()].copy()
    if primary_only:
        include = mapping["include_primary_eval"]
        if include.dtype != bool:
            include = include.astype(str).str.lower().isin({"true", "1", "yes"})
        mapping = mapping[include].copy()

    if mapping.empty:
        raise ValueError("No evaluable mappings remain after filtering.")

    return mapping.reset_index(drop=True)


def prepare_condition(
    pred_path: Path,
    gold_df: pd.DataFrame,
    mapping: pd.DataFrame,
    condition_key: str,
) -> pd.DataFrame:
    pred_df = pd.read_csv(pred_path)

    if "reddit_id" not in pred_df.columns or "reddit_id" not in gold_df.columns:
        raise ValueError("Both prediction and gold files must contain reddit_id.")

    duplicate_ids = pred_df["reddit_id"].duplicated().sum()
    if duplicate_ids:
        raise ValueError(f"{pred_path} contains {duplicate_ids} duplicated reddit_id values.")

    merged = gold_df.merge(
        pred_df,
        on="reddit_id",
        how="inner",
        suffixes=("_goldmeta", "_predmeta"),
        validate="one_to_one",
    )

    if len(merged) != len(gold_df):
        print(
            f"Warning: {condition_key} matched {len(merged)} of {len(gold_df)} CovidET posts."
        )

    return merged


def mapped_arrays(
    df: pd.DataFrame,
    mapping_row: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    crowd_dim = str(mapping_row["crowd_envent_dimension"])
    covidet_dim = str(mapping_row["covidet_column"])
    pred_col = f"{crowd_dim}_pred"
    gold_col = f"covidet_{covidet_dim}_norm"

    missing = [c for c in (pred_col, gold_col) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns for {crowd_dim} <-> {covidet_dim}: {missing}"
        )

    pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    gold = pd.to_numeric(df[gold_col], errors="coerce").to_numpy(dtype=float)

    if str(mapping_row.get("direction", "same")).lower() == "reversed":
        pred = 1.0 - pred

    return pred, gold


def evaluate_dimension(
    df: pd.DataFrame,
    mapping_row: pd.Series,
    condition_key: str,
) -> dict:
    pred, gold = mapped_arrays(df, mapping_row)
    mask = np.isfinite(pred) & np.isfinite(gold)
    pred = pred[mask]
    gold = gold[mask]

    intercept, slope, calibration_r2 = calibration_stats(pred, gold)

    return {
        "condition": condition_key,
        "condition_label": CONDITION_LABELS[condition_key],
        "crowd_envent_dimension": mapping_row["crowd_envent_dimension"],
        "covidet_dimension": mapping_row["covidet_dimension"],
        "covidet_column": mapping_row["covidet_column"],
        "mapping_type": mapping_row.get("mapping_type", ""),
        "mapping_confidence": mapping_row.get("confidence", ""),
        "direction": mapping_row.get("direction", "same"),
        "N": int(len(gold)),
        "gold_mean": float(np.mean(gold)),
        "gold_sd": float(np.std(gold, ddof=1)),
        "prediction_mean": float(np.mean(pred)),
        "prediction_sd": float(np.std(pred, ddof=1)),
        "pearson": safe_corr(pred, gold, "pearson"),
        "spearman": safe_corr(pred, gold, "spearman"),
        "mae": float(mean_absolute_error(gold, pred)),
        "rmse": float(mean_squared_error(gold, pred) ** 0.5),
        "mean_bias_pred_minus_gold": float(np.mean(pred - gold)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_r2": calibration_r2,
    }


def macro_summary(per_dim: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "pearson",
        "spearman",
        "mae",
        "rmse",
        "mean_bias_pred_minus_gold",
        "calibration_slope",
        "calibration_intercept",
    ]
    rows = []
    for condition, group in per_dim.groupby("condition", sort=False):
        row = {
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "n_dimensions": int(len(group)),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"macro_{metric}"] = float(values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def subset_mask(df: pd.DataFrame, subset: str) -> np.ndarray:
    token_col = "token_count_goldmeta" if "token_count_goldmeta" in df.columns else "token_count"
    tokens = pd.to_numeric(df[token_col], errors="coerce").to_numpy()

    if subset == "all":
        return np.ones(len(df), dtype=bool)
    if subset == "gt128":
        return tokens > 128
    if subset == "gt256":
        return tokens > 256
    raise ValueError(f"Unknown subset: {subset}")


def macro_metrics_for_indices(
    condition_df: pd.DataFrame,
    mapping: pd.DataFrame,
    indices: np.ndarray,
) -> Dict[str, float]:
    pearsons: List[float] = []
    spearmans: List[float] = []
    maes: List[float] = []
    rmses: List[float] = []

    sampled = condition_df.iloc[indices]
    for _, row in mapping.iterrows():
        pred, gold = mapped_arrays(sampled, row)
        mask = np.isfinite(pred) & np.isfinite(gold)
        pred = pred[mask]
        gold = gold[mask]
        if len(gold) < 3:
            continue
        pearsons.append(safe_corr(pred, gold, "pearson"))
        spearmans.append(safe_corr(pred, gold, "spearman"))
        maes.append(float(mean_absolute_error(gold, pred)))
        rmses.append(float(mean_squared_error(gold, pred) ** 0.5))

    return {
        "pearson": float(np.nanmean(pearsons)),
        "spearman": float(np.nanmean(spearmans)),
        "mae": float(np.nanmean(maes)),
        "rmse": float(np.nanmean(rmses)),
    }


def aligned_condition_frames(
    frames: Mapping[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    common_ids = None
    for df in frames.values():
        ids = set(df["reddit_id"].astype(str))
        common_ids = ids if common_ids is None else common_ids & ids

    if not common_ids:
        raise ValueError("No reddit_id values are common across all conditions.")

    ordered_ids = sorted(common_ids)
    aligned = {}
    for key, df in frames.items():
        indexed = df.assign(reddit_id=df["reddit_id"].astype(str)).set_index("reddit_id")
        aligned[key] = indexed.loc[ordered_ids].reset_index()
    return aligned


def paired_bootstrap(
    frames: Mapping[str, pd.DataFrame],
    mapping: pd.DataFrame,
    comparisons: Sequence[Tuple[str, str]],
    subsets: Sequence[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    aligned = aligned_condition_frames(frames)
    rng = np.random.default_rng(seed)
    metrics = ("pearson", "spearman", "mae", "rmse")
    rows = []

    reference_df = next(iter(aligned.values()))

    for subset in subsets:
        keep = subset_mask(reference_df, subset)
        base_indices = np.flatnonzero(keep)
        if len(base_indices) < 10:
            print(f"Skipping subset {subset}: only {len(base_indices)} posts.")
            continue

        point_estimates = {
            key: macro_metrics_for_indices(df, mapping, base_indices)
            for key, df in aligned.items()
        }

        for condition_a, condition_b in comparisons:
            boot_diffs = {metric: [] for metric in metrics}

            for _ in range(n_boot):
                sampled = rng.choice(base_indices, size=len(base_indices), replace=True)
                stats_a = macro_metrics_for_indices(aligned[condition_a], mapping, sampled)
                stats_b = macro_metrics_for_indices(aligned[condition_b], mapping, sampled)
                for metric in metrics:
                    boot_diffs[metric].append(stats_a[metric] - stats_b[metric])

            for metric in metrics:
                values = np.asarray(boot_diffs[metric], dtype=float)
                point_diff = (
                    point_estimates[condition_a][metric]
                    - point_estimates[condition_b][metric]
                )
                ci_low, ci_high = np.nanpercentile(values, [2.5, 97.5])

                # For correlations, positive means A is better.
                # For error metrics, negative means A is better.
                if metric in {"mae", "rmse"}:
                    favorable = "negative"
                    improvement = -point_diff
                else:
                    favorable = "positive"
                    improvement = point_diff

                rows.append(
                    {
                        "subset": subset,
                        "n_posts": int(len(base_indices)),
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "comparison": f"{CONDITION_LABELS[condition_a]} − {CONDITION_LABELS[condition_b]}",
                        "metric": metric,
                        "condition_a_value": point_estimates[condition_a][metric],
                        "condition_b_value": point_estimates[condition_b][metric],
                        "difference_a_minus_b": point_diff,
                        "ci_2.5": float(ci_low),
                        "ci_97.5": float(ci_high),
                        "better_direction_for_difference": favorable,
                        "improvement_positive_is_better": improvement,
                        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                        "n_bootstrap": int(n_boot),
                    }
                )

    return pd.DataFrame(rows)


def save_correlation_plot(
    df: pd.DataFrame,
    condition_key: str,
    output_path: Path,
) -> None:
    plot_df = df[df["condition"] == condition_key].copy()
    plot_df = plot_df.sort_values("pearson", ascending=True)

    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(9, max(5.5, 0.42 * len(plot_df))))
    ax.scatter(plot_df["pearson"], y, label="Pearson")
    ax.scatter(plot_df["spearman"], y, marker="s", label="Spearman")
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["covidet_dimension"])
    ax.set_xlabel("Cross-dataset correlation")
    ax.set_title(f"Dimension-level CovidET transfer — {CONDITION_LABELS[condition_key]}")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_mean_calibration_plot(
    df: pd.DataFrame,
    condition_key: str,
    output_path: Path,
) -> None:
    plot_df = df[df["condition"] == condition_key].copy()

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(plot_df["prediction_mean"], plot_df["gold_mean"])

    low = min(plot_df["prediction_mean"].min(), plot_df["gold_mean"].min())
    high = max(plot_df["prediction_mean"].max(), plot_df["gold_mean"].max())
    padding = 0.04
    ax.plot([low - padding, high + padding], [low - padding, high + padding], linestyle="--")

    for _, row in plot_df.iterrows():
        ax.annotate(
            row["covidet_dimension"],
            (row["prediction_mean"], row["gold_mean"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Mean model prediction")
    ax.set_ylabel("Mean CovidET gold rating")
    ax.set_title(f"Cross-dataset scale alignment — {CONDITION_LABELS[condition_key]}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def choose_calibration_dimensions(per_dim: pd.DataFrame, n: int = 4) -> List[str]:
    ranked = per_dim.dropna(subset=["pearson"]).sort_values("pearson")
    if ranked.empty:
        return []
    candidates = [
        ranked.iloc[-1]["crowd_envent_dimension"],  # strongest
        ranked.iloc[len(ranked) // 2]["crowd_envent_dimension"],  # middle
        ranked.iloc[0]["crowd_envent_dimension"],  # weakest
        ranked.iloc[
            np.argmax(np.abs(ranked["mean_bias_pred_minus_gold"].to_numpy()))
        ]["crowd_envent_dimension"],  # largest bias
    ]
    unique = []
    for item in candidates:
        if item not in unique:
            unique.append(item)
    return unique[:n]


def save_selected_calibration_plot(
    condition_df: pd.DataFrame,
    mapping: pd.DataFrame,
    per_dim: pd.DataFrame,
    condition_key: str,
    output_path: Path,
) -> None:
    selected = choose_calibration_dimensions(
        per_dim[per_dim["condition"] == condition_key]
    )
    if not selected:
        return

    n = len(selected)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.5 * rows), squeeze=False)

    for ax, crowd_dim in zip(axes.flat, selected):
        map_row = mapping[mapping["crowd_envent_dimension"] == crowd_dim].iloc[0]
        pred, gold = mapped_arrays(condition_df, map_row)
        mask = np.isfinite(pred) & np.isfinite(gold)
        pred = pred[mask]
        gold = gold[mask]

        ax.scatter(pred, gold, alpha=0.35, s=18)
        if len(pred) >= 3 and np.std(pred) > 0:
            fit = LinearRegression().fit(pred.reshape(-1, 1), gold)
            xs = np.linspace(pred.min(), pred.max(), 100)
            ax.plot(xs, fit.predict(xs.reshape(-1, 1)), linewidth=1.5)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Model prediction")
        ax.set_ylabel("CovidET gold")
        display = map_row["covidet_dimension"]
        result = per_dim[
            (per_dim["condition"] == condition_key)
            & (per_dim["crowd_envent_dimension"] == crowd_dim)
        ].iloc[0]
        ax.set_title(
            f"{display}\nPearson={result['pearson']:.2f}, "
            f"slope={result['calibration_slope']:.2f}, "
            f"bias={result['mean_bias_pred_minus_gold']:+.2f}"
        )
        ax.grid(alpha=0.2)

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Representative CovidET calibration patterns — {CONDITION_LABELS[condition_key]}",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_bootstrap_plot(bootstrap_df: pd.DataFrame, output_path: Path) -> None:
    if bootstrap_df.empty:
        return

    plot_df = bootstrap_df[
        (bootstrap_df["comparison"] == "Sliding window − Head")
        & (bootstrap_df["metric"].isin(["pearson", "mae"]))
    ].copy()
    if plot_df.empty:
        return

    plot_df["label"] = (
        plot_df["subset"]
        .map({"all": "All posts", "gt128": ">128 tokens", "gt256": ">256 tokens"})
        + " — "
        + plot_df["metric"].map({"pearson": "Pearson", "mae": "MAE"})
    )
    plot_df["display_effect"] = np.where(
        plot_df["metric"].isin(["mae", "rmse"]),
        -plot_df["difference_a_minus_b"],
        plot_df["difference_a_minus_b"],
    )
    plot_df["display_low"] = np.where(
        plot_df["metric"].isin(["mae", "rmse"]),
        -plot_df["ci_97.5"],
        plot_df["ci_2.5"],
    )
    plot_df["display_high"] = np.where(
        plot_df["metric"].isin(["mae", "rmse"]),
        -plot_df["ci_2.5"],
        plot_df["ci_97.5"],
    )

    y = np.arange(len(plot_df))
    xerr = np.vstack(
        [
            plot_df["display_effect"] - plot_df["display_low"],
            plot_df["display_high"] - plot_df["display_effect"],
        ]
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(
        plot_df["display_effect"],
        y,
        xerr=xerr,
        fmt="o",
        capsize=4,
    )
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Improvement of sliding window over head-only\n(positive = better)")
    ax.set_title("Paired bootstrap estimates for long-text inference")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--head-tail", dest="head_tail", type=Path, required=True)
    parser.add_argument("--sliding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--include-exploratory-mappings",
        action="store_true",
        help="Include mappings not marked include_primary_eval=True.",
    )
    parser.add_argument(
        "--plot-condition",
        choices=list(CONDITION_LABELS),
        default="sliding",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(
        args.mapping,
        primary_only=not args.include_exploratory_mappings,
    )
    gold = pd.read_csv(args.gold)

    prediction_paths = {
        "head": args.head,
        "tail": args.tail,
        "head_tail": args.head_tail,
        "sliding": args.sliding,
    }
    frames = {
        key: prepare_condition(path, gold, mapping, key)
        for key, path in prediction_paths.items()
    }

    dimension_rows = []
    for condition_key, frame in frames.items():
        for _, mapping_row in mapping.iterrows():
            dimension_rows.append(
                evaluate_dimension(frame, mapping_row, condition_key)
            )

    per_dim = pd.DataFrame(dimension_rows)
    per_dim.to_csv(table_dir / "covidet_dimension_transfer.csv", index=False)

    summary = macro_summary(per_dim)
    summary.to_csv(table_dir / "covidet_condition_macro_summary.csv", index=False)

    comparisons = [
        ("sliding", "head"),
        ("tail", "head"),
        ("head_tail", "head"),
        ("sliding", "tail"),
    ]
    bootstrap = paired_bootstrap(
        frames=frames,
        mapping=mapping,
        comparisons=comparisons,
        subsets=("all", "gt128", "gt256"),
        n_boot=args.n_bootstrap,
        seed=args.bootstrap_seed,
    )
    bootstrap.to_csv(table_dir / "covidet_truncation_bootstrap.csv", index=False)

    condition = args.plot_condition
    save_correlation_plot(
        per_dim,
        condition,
        figure_dir / f"covidet_dimension_correlations_{condition}.png",
    )
    save_mean_calibration_plot(
        per_dim,
        condition,
        figure_dir / f"covidet_gold_vs_prediction_means_{condition}.png",
    )
    save_selected_calibration_plot(
        frames[condition],
        mapping,
        per_dim,
        condition,
        figure_dir / f"covidet_selected_calibration_{condition}.png",
    )
    save_bootstrap_plot(
        bootstrap,
        figure_dir / "covidet_truncation_effects.png",
    )

    metadata = {
        "primary_only": not args.include_exploratory_mappings,
        "n_mappings": int(len(mapping)),
        "n_bootstrap": int(args.n_bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "plot_condition": condition,
        "conditions": list(frames),
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nDimension-level transfer:")
    print(
        per_dim[
            [
                "condition_label",
                "covidet_dimension",
                "pearson",
                "spearman",
                "mae",
                "rmse",
                "mean_bias_pred_minus_gold",
                "calibration_slope",
            ]
        ].round(4).to_string(index=False)
    )
    print("\nMacro summary:")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
