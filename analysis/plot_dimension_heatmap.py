from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path(
    "/workspace/data/results/per_dimension_pearson_table.csv"
)

OUTPUT_PATH = Path(
    "/workspace/data/results/per_dimension_pearson_heatmap.png"
)


df = pd.read_csv(
    INPUT_PATH,
    index_col=0,
)

fig, ax = plt.subplots(figsize=(8, 9))

image = ax.imshow(
    df.values,
    aspect="auto",
)

ax.set_xticks(range(len(df.columns)))
ax.set_xticklabels(
    df.columns,
    rotation=30,
    ha="right",
)

ax.set_yticks(range(len(df.index)))
ax.set_yticklabels(df.index)

ax.set_xlabel("Architecture")
ax.set_ylabel("Appraisal dimension")
ax.set_title("Per-dimension Pearson correlation")

colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label("Pearson correlation")

for row_idx in range(df.shape[0]):
    for col_idx in range(df.shape[1]):
        value = df.iloc[row_idx, col_idx]

        if pd.notna(value):
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

fig.tight_layout()

fig.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved: {OUTPUT_PATH}")