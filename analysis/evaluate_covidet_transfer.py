from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ENSEMBLE_FILES = {
    "head_128": "ensemble_head_128.csv",
    "tail_128": "ensemble_tail_128.csv",
    "head_tail_128": "ensemble_head_tail_128.csv",
    "sliding_window_128_stride64": "ensemble_sliding_window_128_stride64.csv",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("outputs") / "covidet" / "predictions",
    )
    parser.add_argument(
        "--covidet",
        type=Path,
        default=Path("data") / "covidet" / "covidet_posts_preprocessed.csv",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("analysis") / "covidet_dimension_mapping.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "covidet" / "evaluation",
    )
    return parser.parse_args()


def safe_corr(func, gold, pred):
    if len(gold) < 2 or np.std(gold) == 0 or np.std(pred) == 0:
        return np.nan
    value = func(gold, pred).statistic
    return float(value) if np.isfinite(value) else np.nan


def classify_mapping(row):
    mapping_type = str(row["mapping_type"]).strip().lower()
    confidence = str(row["confidence"]).strip().lower()
    direction = str(row["direction"]).strip().lower()
    include_primary = str(row["include_primary_eval"]).strip().lower() in {
        "true", "1", "yes"
    }

    if mapping_type in {"excluded", "unavailable"}:
        return mapping_type
    if include_primary and confidence == "high" and direction == "same":
        return "primary_high_confidence"
    if mapping_type == "approximate":
        return "approximate"
    if mapping_type == "exploratory" and direction == "reversed":
        return "exploratory_reverse"
    return "other_exploratory"


def transform_gold(series, direction):
    values = pd.to_numeric(series, errors="coerce")
    return 1.0 - values if direction == "reversed" else values


def metric_row(gold, pred):
    valid = np.isfinite(gold) & np.isfinite(pred)
    gold = gold[valid]
    pred = pred[valid]

    if len(gold) == 0:
        return {
            "N": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Pearson": np.nan,
            "Spearman": np.nan,
            "Mean_Gold": np.nan,
            "Mean_Pred": np.nan,
            "Mean_Bias": np.nan,
        }

    error = pred - gold
    return {
        "N": len(gold),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "Pearson": safe_corr(pearsonr, gold, pred),
        "Spearman": safe_corr(spearmanr, gold, pred),
        "Mean_Gold": float(np.mean(gold)),
        "Mean_Pred": float(np.mean(pred)),
        "Mean_Bias": float(np.mean(error)),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in [args.covidet, args.mapping]:
        if not path.exists():
            raise FileNotFoundError(path)

    covidet = pd.read_csv(args.covidet)
    mapping = pd.read_csv(args.mapping)
    mapping["Analysis_Set"] = mapping.apply(classify_mapping, axis=1)

    per_dimension_rows = []
    length_rows = []

    for condition, filename in ENSEMBLE_FILES.items():
        prediction_path = args.predictions_dir / filename
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)

        pred_df = pd.read_csv(prediction_path)
        merged = covidet.merge(
            pred_df,
            on="reddit_id",
            how="inner",
            suffixes=("_goldmeta", "_predmeta"),
            validate="one_to_one",
        )

        if len(merged) != len(covidet):
            raise ValueError(
                f"{condition}: matched {len(merged)} of {len(covidet)} posts."
            )

        for _, map_row in mapping.iterrows():
            analysis_set = map_row["Analysis_Set"]
            if analysis_set in {"excluded", "unavailable"}:
                continue

            crowd_dim = str(map_row["crowd_envent_dimension"])
            covidet_column = map_row["covidet_column"]
            if pd.isna(covidet_column):
                continue

            covidet_column = str(covidet_column)
            direction = str(map_row["direction"]).strip().lower()
            gold_col = f"covidet_{covidet_column}_norm"
            pred_col = f"{crowd_dim}_pred"

            if gold_col not in merged.columns:
                raise ValueError(f"Missing gold column: {gold_col}")
            if pred_col not in merged.columns:
                raise ValueError(f"Missing prediction column: {pred_col}")

            gold = transform_gold(merged[gold_col], direction).to_numpy(float)
            pred = pd.to_numeric(merged[pred_col], errors="coerce").to_numpy(float)

            per_dimension_rows.append({
                "Condition": condition,
                "Analysis_Set": analysis_set,
                "Crowd_enVENT_Dimension": crowd_dim,
                "CovidET_Column": covidet_column,
                "CovidET_Dimension": map_row.get("covidet_dimension", np.nan),
                "Mapping_Type": map_row["mapping_type"],
                "Direction": map_row["direction"],
                "Confidence": map_row["confidence"],
                **metric_row(gold, pred),
            })

        primary = mapping[
            mapping["Analysis_Set"] == "primary_high_confidence"
        ]

        length_col = (
            "length_bin_goldmeta"
            if "length_bin_goldmeta" in merged.columns
            else "length_bin"
        )

        for length_bin, bin_df in merged.groupby(
            length_col,
            observed=False,
        ):
            for _, map_row in primary.iterrows():
                crowd_dim = str(map_row["crowd_envent_dimension"])
                covidet_column = str(map_row["covidet_column"])
                direction = str(map_row["direction"]).strip().lower()

                gold = transform_gold(
                    bin_df[f"covidet_{covidet_column}_norm"],
                    direction,
                ).to_numpy(float)

                pred = pd.to_numeric(
                    bin_df[f"{crowd_dim}_pred"],
                    errors="coerce",
                ).to_numpy(float)

                length_rows.append({
                    "Condition": condition,
                    "Length_Bin": length_bin,
                    "Dimension": crowd_dim,
                    **metric_row(gold, pred),
                })

    per_dimension = pd.DataFrame(per_dimension_rows)

    macro = (
        per_dimension
        .groupby(["Condition", "Analysis_Set"], as_index=False)
        .agg(
            N_Dimensions=("Crowd_enVENT_Dimension", "nunique"),
            Macro_RMSE=("RMSE", "mean"),
            Macro_MAE=("MAE", "mean"),
            Macro_Pearson=("Pearson", "mean"),
            Macro_Spearman=("Spearman", "mean"),
            Mean_Absolute_Bias=("Mean_Bias", lambda s: s.abs().mean()),
        )
    )

    length_by_dim = pd.DataFrame(length_rows)
    length_macro = (
        length_by_dim
        .groupby(["Condition", "Length_Bin"], as_index=False, observed=False)
        .agg(
            N_Posts=("N", "max"),
            N_Dimensions=("Dimension", "nunique"),
            Macro_RMSE=("RMSE", "mean"),
            Macro_MAE=("MAE", "mean"),
            Macro_Pearson=("Pearson", "mean"),
            Macro_Spearman=("Spearman", "mean"),
        )
    )

    mapping.to_csv(
        args.output_dir / "covidet_mapping_audit.csv",
        index=False,
    )
    per_dimension.to_csv(
        args.output_dir / "covidet_per_dimension_metrics.csv",
        index=False,
    )
    macro.to_csv(
        args.output_dir / "covidet_macro_metrics_by_mapping_set.csv",
        index=False,
    )
    length_macro.to_csv(
        args.output_dir / "covidet_primary_metrics_by_length_bin.csv",
        index=False,
    )

    print("\nPrimary high-confidence results:")
    print(
        macro[
            macro["Analysis_Set"] == "primary_high_confidence"
        ]
        .sort_values("Macro_Pearson", ascending=False)
        .to_string(index=False)
    )

    print("\nSaved outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
