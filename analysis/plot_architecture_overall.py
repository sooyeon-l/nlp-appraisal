from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path(
    "/workspace/data/results/architecture_comparison.csv"
)

OUTPUT_DIR = Path("/workspace/data/results")


df = pd.read_csv(INPUT_PATH)


# ============================================================
# Macro RMSE
# ============================================================

fig, ax = plt.subplots(figsize=(6.5, 4))

ax.bar(
    df["Model"],
    df["Macro RMSE"],
)

ax.set_ylabel("Macro RMSE")
ax.set_xlabel("Architecture")
ax.set_title("Test macro RMSE by architecture")
ax.tick_params(
    axis="x",
    rotation=20,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "architecture_macro_rmse.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Macro Pearson
# ============================================================

fig, ax = plt.subplots(figsize=(6.5, 4))

ax.bar(
    df["Model"],
    df["Macro Pearson"],
)

ax.set_ylabel("Macro Pearson correlation")
ax.set_xlabel("Architecture")
ax.set_title("Test macro Pearson by architecture")
ax.tick_params(
    axis="x",
    rotation=20,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "architecture_macro_pearson.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved overall architecture figures.")