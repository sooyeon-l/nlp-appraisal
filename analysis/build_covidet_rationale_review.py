from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CONDITION_FILES = {
    "head": "ensemble_head_128.csv",
    "tail": "ensemble_tail_128.csv",
    "head_tail": "ensemble_head_tail_128.csv",
    "sliding": "ensemble_sliding_window_128_stride64.csv",
}

CALIBRATION_DIMS = {
    "goal_relevance",
    "chance_responsblt",
    "chance_control",
    "effort",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a targeted CovidET rationale-inspection dataset "
            "from ensemble predictions and high-confidence mappings."
        )
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
        "--predictions-dir",
        type=Path,
        default=Path("outputs") / "covidet" / "predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "covidet" / "rationale_inspection",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=25,
    )
    return parser.parse_args()


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_primary_mapping(path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(path)

    required = {
        "crowd_envent_dimension",
        "covidet_column",
        "direction",
        "confidence",
        "include_primary_eval",
    }
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(
            "Mapping file is missing columns: "
            + ", ".join(sorted(missing))
        )

    primary = mapping[
        mapping["include_primary_eval"].map(as_bool)
        & mapping["confidence"].astype(str).str.lower().eq("high")
        & mapping["direction"].astype(str).str.lower().eq("same")
    ].copy()

    if primary.empty:
        raise ValueError("No primary high-confidence mappings were found.")

    return primary.reset_index(drop=True)


def load_and_merge_predictions(
    covidet: pd.DataFrame,
    predictions_dir: Path,
) -> pd.DataFrame:
    merged = covidet.copy()

    for short_name, filename in CONDITION_FILES.items():
        path = predictions_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)

        pred = pd.read_csv(path)

        keep = ["reddit_id"] + [
            column
            for column in pred.columns
            if column.endswith("_pred")
        ]

        pred = pred[keep].copy()
        pred = pred.rename(
            columns={
                column: f"{column}_{short_name}"
                for column in pred.columns
                if column != "reddit_id"
            }
        )

        merged = merged.merge(
            pred,
            on="reddit_id",
            how="inner",
            validate="one_to_one",
        )

    if len(merged) != len(covidet):
        raise ValueError(
            f"Expected {len(covidet)} matched posts, found {len(merged)}."
        )

    return merged


def make_candidate_table(
    merged: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, map_row in mapping.iterrows():
        crowd_dim = str(map_row["crowd_envent_dimension"])
        covidet_column = str(map_row["covidet_column"])

        gold_col = f"covidet_{covidet_column}_norm"
        raw_gold_col = f"covidet_{covidet_column}_raw"
        rationale_col = f"covidet_{covidet_column}_rationale"

        required_cols = {
            gold_col,
            rationale_col,
            f"{crowd_dim}_pred_head",
            f"{crowd_dim}_pred_tail",
            f"{crowd_dim}_pred_head_tail",
            f"{crowd_dim}_pred_sliding",
        }
        missing = required_cols - set(merged.columns)
        if missing:
            raise ValueError(
                f"{crowd_dim}: missing columns: "
                + ", ".join(sorted(missing))
            )

        for _, row in merged.iterrows():
            gold = pd.to_numeric(
                pd.Series([row[gold_col]]),
                errors="coerce",
            ).iloc[0]

            if pd.isna(gold):
                continue

            pred_head = float(row[f"{crowd_dim}_pred_head"])
            pred_tail = float(row[f"{crowd_dim}_pred_tail"])
            pred_head_tail = float(row[f"{crowd_dim}_pred_head_tail"])
            pred_sliding = float(row[f"{crowd_dim}_pred_sliding"])

            errors = {
                "head": abs(pred_head - gold),
                "tail": abs(pred_tail - gold),
                "head_tail": abs(pred_head_tail - gold),
                "sliding": abs(pred_sliding - gold),
            }

            rows.append({
                "reddit_id": row["reddit_id"],
                "crowd_envent_dimension": crowd_dim,
                "covidet_column": covidet_column,
                "covidet_dimension": map_row.get(
                    "covidet_dimension",
                    covidet_column,
                ),
                "raw_text": row["raw_text"],
                "rationale": row[rationale_col],
                "character_count": row.get("character_count", np.nan),
                "word_count": row.get("word_count", np.nan),
                "token_count": row.get("token_count", np.nan),
                "length_bin": row.get("length_bin", np.nan),
                "retained_fraction_at_128": row.get(
                    "retained_fraction_at_128",
                    np.nan,
                ),
                "gold_raw": row.get(raw_gold_col, np.nan),
                "gold_norm": float(gold),
                "pred_head": pred_head,
                "pred_tail": pred_tail,
                "pred_head_tail": pred_head_tail,
                "pred_sliding": pred_sliding,
                "abs_error_head": errors["head"],
                "abs_error_tail": errors["tail"],
                "abs_error_head_tail": errors["head_tail"],
                "abs_error_sliding": errors["sliding"],
                "sliding_gain_over_head": (
                    errors["head"] - errors["sliding"]
                ),
                "tail_gain_over_head": (
                    errors["head"] - errors["tail"]
                ),
                "head_tail_gain_over_head": (
                    errors["head"] - errors["head_tail"]
                ),
                "minimum_abs_error": min(errors.values()),
                "maximum_abs_error": max(errors.values()),
                "mean_abs_error": float(np.mean(list(errors.values()))),
                "mean_prediction": float(
                    np.mean([
                        pred_head,
                        pred_tail,
                        pred_head_tail,
                        pred_sliding,
                    ])
                ),
                "mean_signed_error": float(
                    np.mean([
                        pred_head - gold,
                        pred_tail - gold,
                        pred_head_tail - gold,
                        pred_sliding - gold,
                    ])
                ),
            })

    candidates = pd.DataFrame(rows)

    candidates["all_methods_fail"] = (
        candidates["minimum_abs_error"] >= 0.30
    )
    candidates["all_methods_good"] = (
        candidates["maximum_abs_error"] <= 0.15
    )
    candidates["strong_underprediction"] = (
        candidates["mean_signed_error"] <= -0.25
    )
    candidates["long_text"] = (
        candidates["length_bin"].astype(str).eq("257-512")
    )

    return candidates


def add_selection(
    selected: dict[tuple[str, str], dict],
    subset: pd.DataFrame,
    reason: str,
    n: int,
) -> None:
    for _, row in subset.head(n).iterrows():
        key = (
            str(row["reddit_id"]),
            str(row["crowd_envent_dimension"]),
        )

        if key not in selected:
            selected[key] = {
                "row": row,
                "reasons": [],
            }

        if reason not in selected[key]["reasons"]:
            selected[key]["reasons"].append(reason)


def select_review_set(
    candidates: pd.DataFrame,
    target_size: int,
) -> pd.DataFrame:
    selected: dict[tuple[str, str], dict] = {}

    long_sliding = (
        candidates[candidates["long_text"]]
        .sort_values(
            "sliding_gain_over_head",
            ascending=False,
        )
    )
    add_selection(
        selected,
        long_sliding,
        "long_text_sliding_beats_head",
        5,
    )

    tail_wins = candidates.sort_values(
        "tail_gain_over_head",
        ascending=False,
    )
    add_selection(
        selected,
        tail_wins,
        "tail_beats_head",
        5,
    )

    all_fail = (
        candidates[candidates["all_methods_fail"]]
        .sort_values(
            "minimum_abs_error",
            ascending=False,
        )
    )
    add_selection(
        selected,
        all_fail,
        "all_methods_fail",
        5,
    )

    all_good = (
        candidates[candidates["all_methods_good"]]
        .sort_values(
            "maximum_abs_error",
            ascending=True,
        )
    )
    add_selection(
        selected,
        all_good,
        "all_methods_good",
        4,
    )

    calibration = (
        candidates[
            candidates["crowd_envent_dimension"].isin(
                CALIBRATION_DIMS
            )
        ]
        .sort_values(
            "mean_signed_error",
            ascending=True,
        )
    )
    add_selection(
        selected,
        calibration,
        "strong_calibration_underprediction",
        6,
    )

    if len(selected) < target_size:
        fallback = candidates.sort_values(
            "sliding_gain_over_head",
            ascending=False,
        )
        add_selection(
            selected,
            fallback,
            "fallback_large_sliding_gain",
            target_size - len(selected),
        )

    records = []

    for item in selected.values():
        record = item["row"].to_dict()
        record["selection_reasons"] = "; ".join(
            item["reasons"]
        )
        records.append(record)

    review = pd.DataFrame(records)

    if len(review) > target_size:
        review["_priority"] = (
            review["selection_reasons"]
            .str.count(";")
            + 1
        )
        review = (
            review
            .sort_values(
                [
                    "_priority",
                    "sliding_gain_over_head",
                    "minimum_abs_error",
                ],
                ascending=[False, False, False],
            )
            .head(target_size)
            .drop(columns="_priority")
        )

    manual_columns = {
        "evidence_location": "",
        "rationale_supports_gold": "",
        "head_missed_relevant_text": "",
        "tail_captured_relevant_text": "",
        "sliding_captured_distributed_evidence": "",
        "construct_mismatch_possible": "",
        "annotation_unclear": "",
        "qualitative_notes": "",
    }

    for column, default in manual_columns.items():
        review[column] = default

    preferred_order = [
        "selection_reasons",
        "reddit_id",
        "crowd_envent_dimension",
        "covidet_dimension",
        "length_bin",
        "token_count",
        "retained_fraction_at_128",
        "gold_raw",
        "gold_norm",
        "pred_head",
        "pred_tail",
        "pred_head_tail",
        "pred_sliding",
        "abs_error_head",
        "abs_error_tail",
        "abs_error_head_tail",
        "abs_error_sliding",
        "sliding_gain_over_head",
        "tail_gain_over_head",
        "all_methods_fail",
        "all_methods_good",
        "strong_underprediction",
        "rationale",
        "raw_text",
        "evidence_location",
        "rationale_supports_gold",
        "head_missed_relevant_text",
        "tail_captured_relevant_text",
        "sliding_captured_distributed_evidence",
        "construct_mismatch_possible",
        "annotation_unclear",
        "qualitative_notes",
    ]

    remaining = [
        column
        for column in review.columns
        if column not in preferred_order
    ]

    return review[preferred_order + remaining]


def make_selection_summary(review: pd.DataFrame) -> pd.DataFrame:
    counter = defaultdict(int)

    for reasons in review["selection_reasons"].fillna(""):
        for reason in str(reasons).split("; "):
            if reason:
                counter[reason] += 1

    return pd.DataFrame(
        [
            {
                "selection_reason": reason,
                "n_cases": count,
            }
            for reason, count in sorted(counter.items())
        ]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    covidet = pd.read_csv(args.covidet)
    mapping = load_primary_mapping(args.mapping)

    merged = load_and_merge_predictions(
        covidet=covidet,
        predictions_dir=args.predictions_dir,
    )

    candidates = make_candidate_table(
        merged=merged,
        mapping=mapping,
    )

    review = select_review_set(
        candidates=candidates,
        target_size=args.target_size,
    )

    summary = make_selection_summary(review)

    candidates_path = (
        args.output_dir
        / "covidet_rationale_all_candidates.csv"
    )
    review_path = (
        args.output_dir
        / "covidet_rationale_review_set.csv"
    )
    summary_path = (
        args.output_dir
        / "covidet_rationale_selection_summary.csv"
    )

    candidates.to_csv(candidates_path, index=False)
    review.to_csv(review_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Candidate rows: {len(candidates)}")
    print(f"Selected review cases: {len(review)}")
    print()
    print(summary.to_string(index=False))
    print()
    print("Saved:")
    print(candidates_path)
    print(review_path)
    print(summary_path)


if __name__ == "__main__":
    main()
