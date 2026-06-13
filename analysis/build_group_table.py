from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "/workspace/data/results/architecture_comparison.csv"
)

OUTPUT_PATH = Path(
    "/workspace/data/results/group_comparison_table.csv"
)


df = pd.read_csv(INPUT_PATH)

columns = [
    "Model",
    "Relevance RMSE",
    "Relevance Pearson",
    "Implication RMSE",
    "Implication Pearson",
    "Coping RMSE",
    "Coping Pearson",
    "Normative RMSE",
    "Normative Pearson",
]

group_table = df[columns].copy()

group_table.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(group_table.round(4).to_string(index=False))
print(f"\nSaved: {OUTPUT_PATH}")