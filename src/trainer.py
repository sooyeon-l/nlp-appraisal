from __future__ import annotations

import numpy as np
import torch

from src.loss import appraisal_loss


def _pearson_correlation(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    if len(predictions) < 2:
        return float("nan")

    if np.std(predictions) == 0:
        return float("nan")

    if np.std(labels) == 0:
        return float("nan")

    return float(
        np.corrcoef(predictions, labels)[0, 1]
    )


def evaluate_model(
    model,
    dataloader,
    target_dims: list[str],
    objective_groups: dict[str, list[str]],
    loss_mode: str,
    device: str,
) -> dict:
    model.eval()

    all_predictions = []
    all_labels = []
    all_masks = []

    total_objective_loss = 0.0
    total_batches = 0

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

            total_objective_loss += loss.item()
            total_batches += 1

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_labels.append(
                labels.cpu().numpy()
            )

            all_masks.append(
                valid_mask.cpu().numpy()
            )

    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    masks = np.concatenate(all_masks, axis=0).astype(bool)

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

        per_dim_pearson[dim_name] = _pearson_correlation(
            dim_predictions,
            dim_labels,
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

    return {
        "objective_loss": (
            total_objective_loss / max(total_batches, 1)
        ),
        "macro_rmse": float(
            np.nanmean(list(per_dim_rmse.values()))
        ),
        "macro_mae": float(
            np.nanmean(list(per_dim_mae.values()))
        ),
        "macro_pearson": float(
            np.nanmean(list(per_dim_pearson.values()))
        ),
        "per_dim_rmse": per_dim_rmse,
        "per_dim_mae": per_dim_mae,
        "per_dim_pearson": per_dim_pearson,
        "group_metrics": group_metrics,
        "all_predictions": predictions,
        "all_labels": labels,
        "all_masks": masks,
    }