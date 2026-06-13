from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "/workspace/data/results/architecture_comparison.csv"
)

OUTPUT_DIR = Path("/workspace/data/results")


df = pd.read_csv(INPUT_PATH)

groups = [
    "Relevance",
    "Implication",
    "Coping",
    "Normative",
]

x = np.arange(len(groups))
width = 0.18


# ============================================================
# Group Pearson
# ============================================================

fig, ax = plt.subplots(figsize=(8, 4.5))

for idx, row in df.iterrows():
    values = [
        row[f"{group} Pearson"]
        for group in groups
    ]

    ax.bar(
        x + idx * width,
        values,
        width,
        label=row["Model"],
    )

ax.set_xticks(
    x + width * (len(df) - 1) / 2
)

ax.set_xticklabels(groups)

ax.set_ylabel("Mean Pearson correlation")
ax.set_xlabel("CPM objective")
ax.set_title("Architecture performance by CPM objective")
ax.legend()
ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "group_pearson_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Group RMSE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 4.5))

for idx, row in df.iterrows():
    values = [
        row[f"{group} RMSE"]
        for group in groups
    ]

    ax.bar(
        x + idx * width,
        values,
        width,
        label=row["Model"],
    )

ax.set_xticks(
    x + width * (len(df) - 1) / 2
)

ax.set_xticklabels(groups)

ax.set_ylabel("Mean RMSE")
ax.set_xlabel("CPM objective")
ax.set_title("Architecture RMSE by CPM objective")
ax.legend()
ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "group_rmse_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved group comparison figures.")