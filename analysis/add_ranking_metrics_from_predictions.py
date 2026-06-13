from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support

from src.config import TARGET_DIMS


RUNS_ROOT = Path("/workspace/data/runs")

RUNS = [
    "flat_linear_ft_weighted",
    "flat_mlp_ft_weighted",
    "grouped_parallel_ft_weighted",
    "grouped_sequential_ft_weighted",
]

THRESHOLD = 0.75


def calculate_metrics(df: pd.DataFrame) -> dict:
    labels = np.column_stack([
        df[f"{dim}_true"].to_numpy(dtype=float)
        for dim in TARGET_DIMS
    ])

    predictions = np.column_stack([
        df[f"{dim}_pred"].to_numpy(dtype=float)
        for dim in TARGET_DIMS
    ])

    masks = np.isfinite(labels)

    exact_top1 = []
    tie_top1 = []
    top3 = []
    top5 = []
    within_spearman = []

    for pred_row, label_row, mask_row in zip(
        predictions,
        labels,
        masks,
    ):
        valid_indices = np.flatnonzero(mask_row)

        if len(valid_indices) == len(TARGET_DIMS):
            pred_top = int(np.argmax(pred_row))
            gold_top = int(np.argmax(label_row))

            exact_top1.append(pred_top == gold_top)

            gold_max = np.max(label_row)
            tied_gold = np.flatnonzero(
                np.isclose(label_row, gold_max)
            )

            tie_top1.append(pred_top in tied_gold)

        for k, output in [
            (3, top3),
            (5, top5),
        ]:
            if len(valid_indices) < k:
                continue

            pred_valid = pred_row[valid_indices]
            gold_valid = label_row[valid_indices]

            pred_top_k = set(
                valid_indices[
                    np.argsort(pred_valid)[-k:]
                ]
            )

            gold_top_k = set(
                valid_indices[
                    np.argsort(gold_valid)[-k:]
                ]
            )

            output.append(
                len(pred_top_k & gold_top_k) / k
            )

        pred_valid = pred_row[valid_indices]
        gold_valid = label_row[valid_indices]

        if (
            len(valid_indices) >= 3
            and np.std(pred_valid) > 0
            and np.std(gold_valid) > 0
        ):
            rho = spearmanr(
                pred_valid,
                gold_valid,
            ).statistic

            if np.isfinite(rho):
                within_spearman.append(float(rho))

    valid_predictions = predictions[masks]
    valid_labels = labels[masks]

    binary_gold = (
        valid_labels >= THRESHOLD
    ).astype(int)

    binary_predictions = (
        valid_predictions >= THRESHOLD
    ).astype(int)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            binary_gold,
            binary_predictions,
            average="binary",
            zero_division=0,
        )
    )

    return {
        "exact_top1_accuracy": float(
            np.mean(exact_top1)
        ),
        "tie_aware_top1_accuracy": float(
            np.mean(tie_top1)
        ),
        "top3_overlap": float(np.mean(top3)),
        "top5_overlap": float(np.mean(top5)),
        "mean_within_entry_spearman": float(
            np.mean(within_spearman)
        ),
        "high_intensity_threshold": THRESHOLD,
        "high_intensity_precision": float(precision),
        "high_intensity_recall": float(recall),
        "high_intensity_f1": float(f1),
    }


for run_name in RUNS:
    run_dir = RUNS_ROOT / run_name

    predictions_path = (
        run_dir / "test_predictions.csv"
    )

    metrics_path = run_dir / "test_metrics.json"

    if not predictions_path.exists():
        print(
            f"Skipping; predictions missing: "
            f"{predictions_path}"
        )
        continue

    df = pd.read_csv(predictions_path)
    new_metrics = calculate_metrics(df)

    if metrics_path.exists():
        with open(
            metrics_path,
            "r",
            encoding="utf-8",
        ) as file:
            metrics = json.load(file)
    else:
        metrics = {}

    metrics["ranking_metrics"] = {
        key: value
        for key, value in new_metrics.items()
        if not key.startswith("high_intensity")
    }

    metrics["high_intensity_metrics"] = {
        "threshold": THRESHOLD,
        "micro_precision": new_metrics[
            "high_intensity_precision"
        ],
        "micro_recall": new_metrics[
            "high_intensity_recall"
        ],
        "micro_f1": new_metrics[
            "high_intensity_f1"
        ],
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print(run_name)
    print(json.dumps(new_metrics, indent=2))