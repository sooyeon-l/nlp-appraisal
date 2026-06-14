from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


summary_path = (
    OUTPUT_DIR
    / "final_test_dimensions_summary.csv"
)

df = pd.read_csv(summary_path)


# ============================================================
# RMSE plot
# ============================================================

rmse_df = df.sort_values(
    "RMSE_Mean",
    ascending=True,
)

fig, ax = plt.subplots(
    figsize=(8, 8),
)

ax.barh(
    rmse_df["Dimension"],
    rmse_df["RMSE_Mean"],
    xerr=rmse_df["RMSE_SD"],
    capsize=2,
)

ax.set_xlabel("Macro RMSE")
ax.set_ylabel("Appraisal dimension")
ax.set_title(
    "Held-out test RMSE by appraisal dimension\n"
    "CPM sequential, mean ± SD across three seeds"
)

fig.tight_layout()

rmse_path = (
    FIGURE_DIR
    / "final_test_dimension_rmse.png"
)

fig.savefig(
    rmse_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# Pearson plot
# ============================================================

pearson_df = df.sort_values(
    "Pearson_Mean",
    ascending=True,
)

fig, ax = plt.subplots(
    figsize=(8, 8),
)

ax.barh(
    pearson_df["Dimension"],
    pearson_df["Pearson_Mean"],
    xerr=pearson_df["Pearson_SD"],
    capsize=2,
)

ax.set_xlabel("Pearson correlation")
ax.set_ylabel("Appraisal dimension")
ax.set_title(
    "Held-out test correlation by appraisal dimension\n"
    "CPM sequential, mean ± SD across three seeds"
)

fig.tight_layout()

pearson_path = (
    FIGURE_DIR
    / "final_test_dimension_pearson.png"
)

fig.savefig(
    pearson_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# High-intensity F1 plot
# ============================================================

f1_df = (
    df
    .dropna(subset=["High_F1_Mean"])
    .sort_values(
        "High_F1_Mean",
        ascending=True,
    )
)

fig, ax = plt.subplots(
    figsize=(8, 8),
)

ax.barh(
    f1_df["Dimension"],
    f1_df["High_F1_Mean"],
    xerr=f1_df["High_F1_SD"],
    capsize=2,
)

ax.set_xlabel("High-intensity F1")
ax.set_ylabel("Appraisal dimension")
ax.set_title(
    "High-intensity appraisal detection by dimension\n"
    "Threshold = 0.75"
)

fig.tight_layout()

f1_path = (
    FIGURE_DIR
    / "final_test_dimension_high_intensity_f1.png"
)

fig.savefig(
    f1_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


print("Saved:")
print(rmse_path)
print(pearson_path)
print(f1_path)