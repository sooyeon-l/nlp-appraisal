from pathlib import Path

import pandas as pd


COVIDET_ROOT = Path("data") / "covidet"

OUTPUT_ROOT = (
    Path("outputs")
    / "covidet"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

path = (
    COVIDET_ROOT
    / "covidet_posts_preprocessed.csv"
)

df = pd.read_csv(path)

summary = pd.DataFrame([
    {
        "N Posts": len(df),
        "Mean Words": df[
            "word_count"
        ].mean(),
        "Median Words": df[
            "word_count"
        ].median(),
        "Mean Tokens": df[
            "token_count"
        ].mean(),
        "Median Tokens": df[
            "token_count"
        ].median(),
        "Max Tokens": df[
            "token_count"
        ].max(),
        "N Over 128": int(
            df["exceeds_128"].sum()
        ),
        "Percent Over 128": (
            100
            * df["exceeds_128"].mean()
        ),
        "Mean Retained Fraction at 128": (
            df[
                "retained_fraction_at_128"
            ].mean()
        ),
    }
])

bins = (
    df["length_bin"]
    .value_counts(sort=False)
    .rename_axis("Length Bin")
    .reset_index(name="N")
)

bins["Percent"] = (
    100 * bins["N"] / len(df)
)

summary_path = (
    OUTPUT_ROOT
    / "covidet_length_summary.csv"
)

bins_path = (
    OUTPUT_ROOT
    / "covidet_length_bins.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

bins.to_csv(
    bins_path,
    index=False,
)

print(summary.to_string(index=False))
print()
print(bins.to_string(index=False))
print()
print("Saved:", summary_path)
print("Saved:", bins_path)
