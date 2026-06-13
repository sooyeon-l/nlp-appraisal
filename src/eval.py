from __future__ import annotations

import torch


def _safe_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a weighted average only over valid positions.
    """

    effective_weights = weights * mask.to(weights.dtype)

    denominator = effective_weights.sum()

    if denominator.item() == 0:
        return values.sum() * 0.0

    numerator = (
        values * effective_weights
    ).sum()

    return numerator / denominator


def _safe_unweighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid_values = values[mask]

    if valid_values.numel() == 0:
        return values.sum() * 0.0

    return valid_values.mean()


def appraisal_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    valid_mask: torch.Tensor,
    target_dims: list[str],
    objective_groups: dict[str, list[str]],
    loss_mode: str,
) -> torch.Tensor:
    """
    Available modes:

    mse
        Ordinary MSE across all valid target values.

    weighted_mse
        Inverse-frequency-weighted MSE across all valid values.

    group_balanced_mse
        Compute ordinary MSE separately within each CPM objective,
        then give each objective equal weight.

    weighted_group_balanced_mse
        Compute inverse-frequency-weighted MSE within each CPM
        objective, then give each objective equal weight.
    """

    if predictions.shape != labels.shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} does not match "
            f"label shape {labels.shape}."
        )

    squared_error = (predictions - labels) ** 2

    if loss_mode == "mse":
        return _safe_unweighted_mean(
            values=squared_error,
            mask=valid_mask,
        )

    if loss_mode == "weighted_mse":
        return _safe_weighted_mean(
            values=squared_error,
            weights=sample_weights,
            mask=valid_mask,
        )

    if loss_mode not in {
        "group_balanced_mse",
        "weighted_group_balanced_mse",
    }:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    dim_to_idx = {
        dim_name: idx
        for idx, dim_name in enumerate(target_dims)
    }

    group_losses = []

    for group_name, group_dims in objective_groups.items():
        indices = [
            dim_to_idx[dim_name]
            for dim_name in group_dims
        ]

        group_error = squared_error[:, indices]
        group_mask = valid_mask[:, indices]
        group_weights = sample_weights[:, indices]

        if loss_mode == "group_balanced_mse":
            group_loss = _safe_unweighted_mean(
                values=group_error,
                mask=group_mask,
            )
        else:
            group_loss = _safe_weighted_mean(
                values=group_error,
                weights=group_weights,
                mask=group_mask,
            )

        group_losses.append(group_loss)

    return torch.stack(group_losses).mean()