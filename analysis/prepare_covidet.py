from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.config import MODEL_NAME


COVIDET_NAMES = {
    1: "self_responsibility",
    2: "other_responsibility",
    3: "circumstances_responsibility",
    4: "problem_focused_coping",
    5: "goal_relevance",
    6: "attentional_activity",
    7: "emotion_focused_coping",
    8: "self_controllable",
    9: "other_controllable",
    10: "circumstances_controllable",
    11: "predictability",
    12: "threat",
    13: "pleasantness",
    14: "certainty",
    15: "goal_conduciveness",
    16: "fairness",
    17: "future_expectancy",
    18: "social_norms",
    19: "loss",
    20: "familiarity",
    21: "effort",
    22: "challenge",
    23: "internal_values",
    24: "expectedness",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "covidet",
    )

    return parser.parse_args()


def normalize_rating(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return (numeric - 1.0) / 8.0


def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = pd.read_csv(args.input)

    print("Shape:", raw.shape)
    print("Columns:")
    print(raw.columns.tolist())

    required = {
        "Reddit ID",
        "Reddit Post",
        *[f"dim{i}" for i in range(1, 25)],
        *[
            f"dim{i}_rationale"
            for i in range(1, 25)
        ],
    }

    missing = sorted(
        required - set(raw.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    output = pd.DataFrame({
        "reddit_id": raw["Reddit ID"],
        "raw_text": raw["Reddit Post"].astype(str),
    })

    for optional_column, output_name in [
        ("HIT ID", "hit_id"),
        ("Assignment ID", "assignment_id"),
        ("Worker ID", "worker_id"),
    ]:
        if optional_column in raw.columns:
            output[output_name] = raw[
                optional_column
            ]

    output["character_count"] = (
        output["raw_text"].str.len()
    )

    output["word_count"] = (
        output["raw_text"]
        .str.split()
        .str.len()
    )

    output["token_count"] = output[
        "raw_text"
    ].map(
        lambda text: len(
            tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
    )

    output["exceeds_128"] = (
        output["token_count"] > 128
    )

    output["retained_fraction_at_128"] = (
        np.minimum(
            1.0,
            128.0 / output["token_count"],
        )
    )

    output["length_bin"] = pd.cut(
        output["token_count"],
        bins=[
            0,
            128,
            256,
            512,
            np.inf,
        ],
        labels=[
            "<=128",
            "129-256",
            "257-512",
            ">512",
        ],
        include_lowest=True,
    )

    for dim_id, readable_name in (
        COVIDET_NAMES.items()
    ):
        raw_col = f"dim{dim_id}"
        rationale_col = (
            f"dim{dim_id}_rationale"
        )

        output[
            f"covidet_dim{dim_id}_raw"
        ] = pd.to_numeric(
            raw[raw_col],
            errors="coerce",
        )

        output[
            f"covidet_dim{dim_id}_norm"
        ] = normalize_rating(
            raw[raw_col]
        )

        output[
            f"covidet_dim{dim_id}_rationale"
        ] = raw[rationale_col]

        output[
            f"covidet_dim{dim_id}_name"
        ] = readable_name

    annotator_path = (
        args.output_dir
        / "covidet_annotator_rows.csv"
    )

    output.to_csv(
        annotator_path,
        index=False,
    )

    aggregation = {
        "raw_text": "first",
        "character_count": "first",
        "word_count": "first",
        "token_count": "first",
        "exceeds_128": "first",
        "retained_fraction_at_128": "first",
        "length_bin": "first",
    }

    for dim_id in COVIDET_NAMES:
        aggregation[
            f"covidet_dim{dim_id}_raw"
        ] = "mean"

        aggregation[
            f"covidet_dim{dim_id}_norm"
        ] = "mean"

    post_level = (
        output
        .groupby(
            "reddit_id",
            as_index=False,
        )
        .agg(aggregation)
    )

    for dim_id in COVIDET_NAMES:
        rationale_col = (
            f"covidet_dim{dim_id}_rationale"
        )

        rationale_lists = (
            output
            .groupby("reddit_id")[
                rationale_col
            ]
            .apply(
                lambda values: [
                    str(value)
                    for value in values
                    if pd.notna(value)
                    and str(value).strip()
                ]
            )
        )

        post_level[
            rationale_col
        ] = post_level["reddit_id"].map(
            rationale_lists
        )

    pickle_path = (
        args.output_dir
        / "covidet_posts_preprocessed.pkl"
    )

    post_level.to_pickle(
        pickle_path
    )

    csv_path = (
        args.output_dir
        / "covidet_posts_preprocessed.csv"
    )

    flat_output = post_level.copy()

    for dim_id in COVIDET_NAMES:
        rationale_col = (
            f"covidet_dim{dim_id}_rationale"
        )

        flat_output[
            rationale_col
        ] = flat_output[
            rationale_col
        ].map(
            lambda values: " ||| ".join(
                values
            )
        )

    flat_output.to_csv(
        csv_path,
        index=False,
    )

    print()
    print("Annotator rows:", len(output))
    print(
        "Unique posts:",
        len(post_level),
    )
    print()
    print(
        post_level[
            [
                "token_count",
                "word_count",
                "retained_fraction_at_128",
            ]
        ].describe()
    )
    print()
    print("Length bins:")
    print(
        post_level["length_bin"]
        .value_counts(sort=False)
    )
    print()
    print("Saved:", annotator_path)
    print("Saved:", pickle_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
