from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support

from src.loss import appraisal_loss


def _pearson_correlation(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    if len(predictions) < 2:
        return float("nan")

    if np.std(predictions) == 0 or np.std(labels) == 0:
        return float("nan")

    return float(np.corrcoef(predictions, labels)[0, 1])


def _exact_top1_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
) -> float:
    """
    Exact agreement between the highest predicted and highest gold
    appraisal dimension.

    Samples without all 21 valid labels are skipped because missing
    dimensions could change the true ranking.
    """
    scores = []

    for pred_row, label_row, mask_row in zip(
        predictions,
        labels,
        masks,
    ):
        if not mask_row.all():
            continue

        pred_top = int(np.argmax(pred_row))
        true_top = int(np.argmax(label_row))

        scores.append(pred_top == true_top)

    return float(np.mean(scores)) if scores else float("nan")


def _tie_aware_top1_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    tolerance: float = 1e-8,
) -> float:
    """
    Counts the prediction as correct when the predicted top dimension
    belongs to the set of dimensions tied for the highest gold score.
    """
    scores = []

    for pred_row, label_row, mask_row in zip(
        predictions,
        labels,
        masks,
    ):
        if not mask_row.all():
            continue

        pred_top = int(np.argmax(pred_row))
        gold_max = np.max(label_row)

        gold_top_indices = np.flatnonzero(
            np.isclose(
                label_row,
                gold_max,
                atol=tolerance,
            )
        )

        scores.append(pred_top in gold_top_indices)

    return float(np.mean(scores)) if scores else float("nan")


def _mean_top_k_overlap(
    predictions: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    k: int,
) -> float:
    """
    Mean proportion of the gold top-k dimensions recovered in the
    predicted top-k dimensions.
    """
    overlaps = []

    for pred_row, label_row, mask_row in zip(
        predictions,
        labels,
        masks,
    ):
        valid_indices = np.flatnonzero(mask_row)

        if len(valid_indices) < k:
            continue

        valid_predictions = pred_row[valid_indices]
        valid_labels = label_row[valid_indices]

        pred_local_top = np.argsort(valid_predictions)[-k:]
        gold_local_top = np.argsort(valid_labels)[-k:]

        pred_top = set(valid_indices[pred_local_top])
        gold_top = set(valid_indices[gold_local_top])

        overlaps.append(len(pred_top & gold_top) / k)

    return float(np.mean(overlaps)) if overlaps else float("nan")


def _mean_within_entry_spearman(
    predictions: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
) -> float:
    """
    For each entry, correlate the predicted and gold rankings across
    appraisal dimensions, then average across entries.

    This differs from per-dimension Pearson, which evaluates one
    appraisal dimension across all entries.
    """
    correlations = []

    for pred_row, label_row, mask_row in zip(
        predictions,
        labels,
        masks,
    ):
        valid_predictions = pred_row[mask_row]
        valid_labels = label_row[mask_row]

        if len(valid_labels) < 3:
            continue

        if (
            np.std(valid_predictions) == 0
            or np.std(valid_labels) == 0
        ):
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = spearmanr(
                valid_predictions,
                valid_labels,
            )

        correlation = float(result.statistic)

        if np.isfinite(correlation):
            correlations.append(correlation)

    return (
        float(np.mean(correlations))
        if correlations
        else float("nan")
    )


def _high_intensity_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    target_dims: list[str],
    threshold: float,
) -> dict:
    """
    Treat appraisal values at or above `threshold` as high-intensity.

    For normalized 1-5 ratings:
        0.00 -> rating 1
        0.25 -> rating 2
        0.50 -> rating 3
        0.75 -> rating 4
        1.00 -> rating 5

    Thus 0.75 corresponds approximately to ratings 4-5.
    """
    valid_predictions = predictions[masks]
    valid_labels = labels[masks]

    gold_binary = (
        valid_labels >= threshold
    ).astype(int)

    pred_binary = (
        valid_predictions >= threshold
    ).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_binary,
        pred_binary,
        average="binary",
        zero_division=0,
    )

    per_dimensional = {}

    for dim_idx, dim_name in enumerate(target_dims):
        valid = masks[:, dim_idx]

        if not valid.any():
            per_dimensional[dim_name] = {
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "support": 0,
            }
            continue

        dim_gold = (
            labels[valid, dim_idx] >= threshold
        ).astype(int)

        dim_pred = (
            predictions[valid, dim_idx] >= threshold
        ).astype(int)

        dim_precision, dim_recall, dim_f1, _ = (
            precision_recall_fscore_support(
                dim_gold,
                dim_pred,
                average="binary",
                zero_division=0,
            )
        )

        per_dimensional[dim_name] = {
            "precision": float(dim_precision),
            "recall": float(dim_recall),
            "f1": float(dim_f1),
            "support": int(dim_gold.sum()),
        }

    return {
        "threshold": float(threshold),
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(f1),
        "per_dimension": per_dimensional,
    }


def evaluate_model(
    model,
    dataloader,
    target_dims: list[str],
    objective_groups: dict[str, list[str]],
    loss_mode: str,
    device: str,
    high_intensity_threshold: float = 0.75,
) -> dict:
    model.eval()

    all_predictions = []
    all_labels = []
    all_masks = []

    total_objective_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            predictions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = appraisal_loss(
                predictions=predictions,
                labels=labels,
                sample_weights=weights,
                valid_mask=valid_mask,
                target_dims=target_dims,
                objective_groups=objective_groups,
                loss_mode=loss_mode,
            )

            batch_size = labels.shape[0]

            total_objective_loss += (
                float(loss.item()) * batch_size
            )
            total_examples += batch_size

            all_predictions.append(
                predictions.detach().cpu().numpy()
            )
            all_labels.append(
                labels.detach().cpu().numpy()
            )
            all_masks.append(
                valid_mask.detach().cpu().numpy()
            )

    predictions = np.concatenate(
        all_predictions,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    masks = np.concatenate(
        all_masks,
        axis=0,
    ).astype(bool)

    per_dim_rmse = {}
    per_dim_mae = {}
    per_dim_pearson = {}

    for dim_idx, dim_name in enumerate(target_dims):
        valid = masks[:, dim_idx]

        dim_predictions = predictions[valid, dim_idx]
        dim_labels = labels[valid, dim_idx]

        if len(dim_labels) == 0:
            per_dim_rmse[dim_name] = float("nan")
            per_dim_mae[dim_name] = float("nan")
            per_dim_pearson[dim_name] = float("nan")
            continue

        errors = dim_predictions - dim_labels

        per_dim_rmse[dim_name] = float(
            np.sqrt(np.mean(errors ** 2))
        )

        per_dim_mae[dim_name] = float(
            np.mean(np.abs(errors))
        )

        per_dim_pearson[dim_name] = (
            _pearson_correlation(
                dim_predictions,
                dim_labels,
            )
        )

    group_metrics = {}

    for group_name, group_dims in objective_groups.items():
        group_metrics[group_name] = {
            "mean_rmse": float(
                np.nanmean([
                    per_dim_rmse[dim]
                    for dim in group_dims
                ])
            ),
            "mean_mae": float(
                np.nanmean([
                    per_dim_mae[dim]
                    for dim in group_dims
                ])
            ),
            "mean_pearson": float(
                np.nanmean([
                    per_dim_pearson[dim]
                    for dim in group_dims
                ])
            ),
        }

    ranking_metrics = {
        "exact_top1_accuracy": _exact_top1_accuracy(
            predictions,
            labels,
            masks,
        ),
        "tie_aware_top1_accuracy": (
            _tie_aware_top1_accuracy(
                predictions,
                labels,
                masks,
            )
        ),
        "top3_overlap": _mean_top_k_overlap(
            predictions,
            labels,
            masks,
            k=3,
        ),
        "top5_overlap": _mean_top_k_overlap(
            predictions,
            labels,
            masks,
            k=5,
        ),
        "mean_within_entry_spearman": (
            _mean_within_entry_spearman(
                predictions,
                labels,
                masks,
            )
        ),
    }

    high_intensity_metrics = _high_intensity_metrics(
        predictions=predictions,
        labels=labels,
        masks=masks,
        target_dims=target_dims,
        threshold=high_intensity_threshold,
    )

    return {
        "objective_loss": (
            total_objective_loss
            / max(total_examples, 1)
        ),
        "macro_rmse": float(
            np.nanmean(list(per_dim_rmse.values()))
        ),
        "macro_mae": float(
            np.nanmean(list(per_dim_mae.values()))
        ),
        "macro_pearson": float(
            np.nanmean(
                list(per_dim_pearson.values())
            )
        ),
        "per_dim_rmse": per_dim_rmse,
        "per_dim_mae": per_dim_mae,
        "per_dim_pearson": per_dim_pearson,
        "group_metrics": group_metrics,
        "ranking_metrics": ranking_metrics,
        "high_intensity_metrics": (
            high_intensity_metrics
        ),
        "all_predictions": predictions,
        "all_labels": labels,
        "all_masks": masks,
    }