from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BRANCHES = [
    "emotion_only",
    "appraisal_only",
    "integrated",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs")
        / "emotion_branch"
        / "emotion_branch_comparison.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs")
        / "emotion_branch"
        / "emotion_branch_manual_summary.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    rows = []
    for branch in BRANCHES:
        relevance = pd.to_numeric(
            df[f"{branch}_relevance_1to5"],
            errors="coerce",
        )
        specificity = pd.to_numeric(
            df[f"{branch}_specificity_1to5"],
            errors="coerce",
        )

        rows.append({
            "branch": branch,
            "n_relevance_rated": int(relevance.notna().sum()),
            "mean_relevance": relevance.mean(),
            "n_specificity_rated": int(specificity.notna().sum()),
            "mean_specificity": specificity.mean(),
            "times_selected_best": int(
                df["best_branch"]
                .fillna("")
                .str.lower()
                .eq(branch)
                .sum()
            ),
        })

    summary = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)

    print(summary.to_string(index=False))
    print()
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
