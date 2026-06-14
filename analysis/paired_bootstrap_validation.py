from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support

from src.config import TARGET_DIMS


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experiment configuration
# ============================================================

SEEDS = [42, 123, 456]

ARCHITECTURES = {
    "Flat linear": "flat_linear",
    "Flat MLP": "flat_mlp",
    "CPM parallel": "grouped_parallel",
    "CPM sequential": "grouped_sequential",
}

COMPARISONS = [
    ("CPM sequential", "Flat linear"),
    ("CPM sequential", "Flat MLP"),
    ("CPM sequential", "CPM parallel"),
]

N_BOOTSTRAPS = 5000
RANDOM_SEED = 2026
HIGH_INTENSITY_THRESHOLD = 0.75


# ============================================================
# Loading
# ============================================================

def prediction_path(
    architecture_tag: str,
    seed: int,
) -> Path:
    run_name = (
        f"{architecture_tag}_ft_mse_seed{seed}"
    )

    return (
        RUNS_ROOT
        / run_name
        / "val_predictions_recomputed.csv"
    )


def load_prediction_arrays(
    architecture_tag: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = prediction_path(
        architecture_tag=architecture_tag,
        seed=seed,
    )

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    labels = np.column_stack([
        df[f"{dimension}_true"].to_numpy(dtype=float)
        for dimension in TARGET_DIMS
    ])

    predictions = np.column_stack([
        df[f"{dimension}_pred"].to_numpy(dtype=float)
        for dimension in TARGET_DIMS
    ])

    valid_mask = np.isfinite(labels)

    return predictions, labels, valid_mask


def load_architecture_ensemble(
    architecture_tag: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed_predictions = []
    reference_labels = None
    reference_mask = None

    for seed in SEEDS:
        predictions, labels, valid_mask = (
            load_prediction_arrays(
                architecture_tag=architecture_tag,
                seed=seed,
            )
        )

        if reference_labels is None:
            reference_labels = labels
            reference_mask = valid_mask
        else:
            if not np.array_equal(
                valid_mask,
                reference_mask,
            ):
                raise ValueError(
                    f"Mask mismatch for "
                    f"{architecture_tag}, seed {seed}"
                )

            if not np.allclose(
                labels[valid_mask],
                reference_labels[valid_mask],
                atol=1e-8,
            ):
                raise ValueError(
                    f"Gold-label mismatch for "
                    f"{architecture_tag}, seed {seed}"
                )

        seed_predictions.append(predictions)

    ensemble_predictions = np.mean(
        np.stack(seed_predictions, axis=0),
        axis=0,
    )

    return (
        ensemble_predictions,
        reference_labels,
        reference_mask,
    )


# ============================================================
# Metrics
# ============================================================

def macro_rmse(
    predictions: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    values = []

    for dim_idx in range(labels.shape[1]):
        valid = valid_mask[:, dim_idx]

        if not valid.any():
            continue

        error = (
            predictions[valid, dim_idx]
            - labels[valid, dim_idx]
        )

        values.append(
            np.sqrt(np.mean(error ** 2))
        )

    return float(np.mean(values))


def macro_mae(
    predictions: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    values = []

    for dim_idx in range(labels.shape[1]):
        valid = valid_mask[:, dim_idx]

        if not valid.any():
            continue

        error = np.abs(
            predictions[valid, dim_idx]
            - labels[valid, dim_idx]
        )

        values.append(np.mean(error))

    return float(np.mean(values))


def macro_pearson(
    predictions: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    values = []

    for dim_idx in range(labels.shape[1]):
        valid = valid_mask[:, dim_idx]

        pred = predictions[valid, dim_idx]
        gold = labels[valid, dim_idx]

        if len(gold) < 2:
            continue

        if np.std(pred) == 0 or np.std(gold) == 0:
            continue

        correlation = np.corrcoef(
            pred,
            gold,
        )[0, 1]

        if np.isfinite(correlation):
            values.append(correlation)

    return float(np.mean(values))


def within_entry_spearman(
    predictions: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    values = []

    for pred_row, gold_row, mask_row in zip(
        predictions,
        labels,
        valid_mask,
    ):
        pred = pred_row[mask_row]
        gold = gold_row[mask_row]

        if len(gold) < 3:
            continue

        if np.std(pred) == 0 or np.std(gold) == 0:
            continue

        correlation = spearmanr(
            pred,
            gold,
        ).statistic

        if np.isfinite(correlation):
            values.append(correlation)

    return float(np.mean(values))


def top_k_overlap(
    predictions: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    k: int,
) -> float:
    values = []

    for pred_row, gold_row, mask_row in zip(
        predictions,
        labels,
        valid_mask,
    ):
        valid_indices = np.flatnonzero(mask_row)

        if len(valid_indices) < k:
            continue

        pred_valid = pred_row[valid_indices]
        gold_valid = gold_row[valid_indices]

        pred_top = set(
            valid_indices[
                np.argsort(pred_valid)[-k:]
            ]
        )

        gold_top = set(
            valid_indices[
                np.argsort(gold_valid)[-k:]
            ]
        )

        values.append(
            len(pred_top & gold_top) / k
        )

    return float(np.mean(values))


def high_intensity_f1(
    predictions: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = HIGH_INTENSITY_THRESHOLD,
) -> float:
    pred_binary = (
        predictions[valid_mask] >= threshold
    ).astype(int)

    gold_binary = (
        labels[valid_mask] >= threshold
    ).astype(int)

    _, _, f1, _ = (
        precision_recall_fscore_support(
            gold_binary,
            pred_binary,
            average="binary",
            zero_division=0,
        )
    )

    return float(f1)


METRICS = {
    "Macro RMSE": macro_rmse,
    "Macro MAE": macro_mae,
    "Macro Pearson": macro_pearson,
    "Within-entry Spearman": within_entry_spearman,
    "Top-3 Overlap": lambda p, y, m: top_k_overlap(
        p,
        y,
        m,
        k=3,
    ),
    "High-intensity F1": high_intensity_f1,
}

LOWER_IS_BETTER = {
    "Macro RMSE",
    "Macro MAE",
}


# ============================================================
# Paired bootstrap
# ============================================================

def paired_bootstrap(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    metric_name: str,
    n_bootstraps: int = N_BOOTSTRAPS,
    random_seed: int = RANDOM_SEED,
) -> dict:
    metric_function = METRICS[metric_name]

    score_a = metric_function(
        predictions_a,
        labels,
        valid_mask,
    )

    score_b = metric_function(
        predictions_b,
        labels,
        valid_mask,
    )

    if metric_name in LOWER_IS_BETTER:
        observed_difference = score_b - score_a
    else:
        observed_difference = score_a - score_b

    rng = np.random.default_rng(random_seed)
    n_samples = labels.shape[0]

    bootstrap_differences = []

    for _ in range(n_bootstraps):
        indices = rng.integers(
            low=0,
            high=n_samples,
            size=n_samples,
        )

        boot_labels = labels[indices]
        boot_mask = valid_mask[indices]

        boot_a = predictions_a[indices]
        boot_b = predictions_b[indices]

        boot_score_a = metric_function(
            boot_a,
            boot_labels,
            boot_mask,
        )

        boot_score_b = metric_function(
            boot_b,
            boot_labels,
            boot_mask,
        )

        if not (
            np.isfinite(boot_score_a)
            and np.isfinite(boot_score_b)
        ):
            continue

        if metric_name in LOWER_IS_BETTER:
            difference = (
                boot_score_b - boot_score_a
            )
        else:
            difference = (
                boot_score_a - boot_score_b
            )

        bootstrap_differences.append(difference)

    bootstrap_differences = np.asarray(
        bootstrap_differences,
        dtype=float,
    )

    ci_low, ci_high = np.percentile(
        bootstrap_differences,
        [2.5, 97.5],
    )

    probability_better = float(
        np.mean(bootstrap_differences > 0)
    )

    two_sided_p = float(
        2
        * min(
            np.mean(bootstrap_differences <= 0),
            np.mean(bootstrap_differences >= 0),
        )
    )

    two_sided_p = min(two_sided_p, 1.0)

    return {
        "Metric": metric_name,
        "Model A Score": score_a,
        "Model B Score": score_b,
        "Difference Favoring A": observed_difference,
        "95% CI Low": ci_low,
        "95% CI High": ci_high,
        "Probability A Better": probability_better,
        "Bootstrap p-value": two_sided_p,
        "Bootstrap Samples": len(
            bootstrap_differences
        ),
    }


# ============================================================
# Load ensemble predictions
# ============================================================

architecture_data = {}

for architecture_label, architecture_tag in (
    ARCHITECTURES.items()
):
    architecture_data[architecture_label] = (
        load_architecture_ensemble(
            architecture_tag=architecture_tag,
        )
    )

    print(
        f"Loaded ensemble predictions: "
        f"{architecture_label}"
    )


# Verify that all architectures use the same labels.
reference_labels = architecture_data[
    "CPM sequential"
][1]

reference_mask = architecture_data[
    "CPM sequential"
][2]

for architecture_label, (
    _,
    labels,
    mask,
) in architecture_data.items():
    if not np.array_equal(mask, reference_mask):
        raise ValueError(
            f"Mask mismatch: {architecture_label}"
        )

    if not np.allclose(
        labels[mask],
        reference_labels[mask],
        atol=1e-8,
    ):
        raise ValueError(
            f"Gold-label mismatch: "
            f"{architecture_label}"
        )


# ============================================================
# Run comparisons
# ============================================================

results = []

for model_a_name, model_b_name in COMPARISONS:
    predictions_a = architecture_data[
        model_a_name
    ][0]

    predictions_b = architecture_data[
        model_b_name
    ][0]

    print()
    print("=" * 80)
    print(f"{model_a_name} vs {model_b_name}")
    print("=" * 80)

    for metric_name in METRICS:
        result = paired_bootstrap(
            predictions_a=predictions_a,
            predictions_b=predictions_b,
            labels=reference_labels,
            valid_mask=reference_mask,
            metric_name=metric_name,
        )

        result["Model A"] = model_a_name
        result["Model B"] = model_b_name

        results.append(result)

        print(
            f"{metric_name:25} "
            f"diff={result['Difference Favoring A']:+.6f} "
            f"CI=[{result['95% CI Low']:+.6f}, "
            f"{result['95% CI High']:+.6f}] "
            f"P(A better)="
            f"{result['Probability A Better']:.3f}"
        )


results_df = pd.DataFrame(results)

column_order = [
    "Model A",
    "Model B",
    "Metric",
    "Model A Score",
    "Model B Score",
    "Difference Favoring A",
    "95% CI Low",
    "95% CI High",
    "Probability A Better",
    "Bootstrap p-value",
    "Bootstrap Samples",
]

results_df = results_df[column_order]

output_path = (
    OUTPUT_DIR
    / "paired_bootstrap_validation_ensemble.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)

print()
print("Saved:", output_path)
print()
print(results_df.to_string(index=False))