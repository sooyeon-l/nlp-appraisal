from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS_ROOT = Path("/workspace/data/runs")
OUTPUT_DIR = Path("/workspace/data/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Change this after you identify the best model.
BEST_RUN = "grouped_parallel_ft_weighted"


PAPER_RMSE = {
    "suddenness": 1.33,
    "familiarity": 1.42,
    "predict_event": 1.47,
    "pleasantness": 1.30,
    "unpleasantness": 1.26,
    "goal_relevance": 1.57,
    "chance_responsblt": 1.43,
    "self_responsblt": 1.40,
    "other_responsblt": 1.57,
    "predict_conseq": 1.50,
    "goal_support": 1.33,
    "urgency": 1.43,
    "self_control": 1.35,
    "other_control": 1.36,
    "chance_control": 1.35,
    "accept_conseq": 1.36,
    "standards": 1.34,
    "social_norms": 1.44,
    "attention": 1.27,
    "not_consider": 1.53,
    "effort": 1.38,
}


metrics_path = (
    RUNS_ROOT
    / BEST_RUN
    / "test_metrics.json"
)

with open(
    metrics_path,
    "r",
    encoding="utf-8",
) as file:
    metrics = json.load(file)


rows = []

for dim_name, paper_rmse in PAPER_RMSE.items():
    your_rmse_normalized = (
        metrics["per_dim_rmse"][dim_name]
    )

    your_rmse_original_scale = (
        your_rmse_normalized * 4.0
    )

    rows.append({
        "Dimension": dim_name,
        "Your RMSE normalized": your_rmse_normalized,
        "Your RMSE 1–5 scale": your_rmse_original_scale,
        "Paper RMSE 1–5 scale": paper_rmse,
        "Difference": (
            your_rmse_original_scale - paper_rmse
        ),
    })


comparison_df = pd.DataFrame(rows)

comparison_df.to_csv(
    OUTPUT_DIR / "paper_dimension_comparison.csv",
    index=False,
)


comparison_df = comparison_df.sort_values(
    "Your RMSE 1–5 scale",
    ascending=True,
).reset_index(drop=True)


y = np.arange(len(comparison_df))

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(
    comparison_df["Paper RMSE 1–5 scale"],
    y,
    label="Original paper",
    marker="o",
)

ax.scatter(
    comparison_df["Your RMSE 1–5 scale"],
    y,
    label="Your model",
    marker="x",
)

for idx, row in comparison_df.iterrows():
    ax.plot(
        [
            row["Paper RMSE 1–5 scale"],
            row["Your RMSE 1–5 scale"],
        ],
        [idx, idx],
        linewidth=0.8,
    )

ax.set_yticks(y)
ax.set_yticklabels(
    comparison_df["Dimension"]
)

ax.set_xlabel("RMSE on original 1–5 scale")
ax.set_ylabel("Appraisal dimension")
ax.set_title(
    "Reference comparison with original crowd-enVENT model"
)

ax.legend()
ax.grid(
    axis="x",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "paper_rmse_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved paper comparison table and figure.")