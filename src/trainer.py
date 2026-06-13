from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.eval import evaluate_model
from src.loss import appraisal_loss


def _json_safe(value):
    if isinstance(value, dict):
        return {
            key: _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    if hasattr(value, "item"):
        return value.item()

    return value


def train_model(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    n_epochs: int,
    patience: int,
    checkpoint_path: str | Path,
    training_log_path: str | Path,
    target_dims: list[str],
    objective_groups: dict[str, list[str]],
    loss_mode: str,
    device: str,
    grad_clip: float = 1.0,
) -> list[float]:
    checkpoint_path = Path(checkpoint_path)
    training_log_path = Path(training_log_path)

    batch_loss_record = []
    epoch_rows = []

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, n_epochs + 1):
        # ====================================================
        # Training
        # ====================================================

        model.train()
        epoch_train_losses = []

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{n_epochs}",
            leave=False,
        )

        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)

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

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

            optimizer.step()

            loss_value = float(loss.item())

            epoch_train_losses.append(loss_value)
            batch_loss_record.append(loss_value)

            progress.set_postfix(
                train_loss=f"{loss_value:.4f}"
            )

        avg_train_loss = sum(epoch_train_losses) / max(
            len(epoch_train_losses),
            1,
        )

        # ====================================================
        # Validation
        # ====================================================

        val_results = evaluate_model(
            model=model,
            dataloader=val_loader,
            target_dims=target_dims,
            objective_groups=objective_groups,
            loss_mode=loss_mode,
            device=device,
        )
        ranking = val_results["ranking_metrics"]
        high_intensity = val_results["high_intensity_metrics"]

        epoch_row.update({
            "exact_top1_accuracy":
                ranking["exact_top1_accuracy"],

            "tie_aware_top1_accuracy":
                ranking["tie_aware_top1_accuracy"],

            "top3_overlap":
                ranking["top3_overlap"],

            "top5_overlap":
                ranking["top5_overlap"],

            "mean_within_entry_spearman":
                ranking["mean_within_entry_spearman"],

            "high_intensity_precision":
                high_intensity["micro_precision"],

            "high_intensity_recall":
                high_intensity["micro_recall"],

            "high_intensity_f1":
                high_intensity["micro_f1"],
        })
        val_objective_loss = val_results["objective_loss"]

        scheduler.step(val_objective_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{n_epochs} | "
            f"train objective={avg_train_loss:.4f} | "
            f"val objective={val_objective_loss:.4f} | "
            f"macro RMSE={val_results['macro_rmse']:.4f} | "
            f"macro r={val_results['macro_pearson']:.4f} | "
            f"within-entry rho="
            f"{ranking['mean_within_entry_spearman']:.4f} | "
            f"top-3 overlap={ranking['top3_overlap']:.4f} | "
            f"high-F1={high_intensity['micro_f1']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # ====================================================
        # Save epoch log
        # ====================================================

        epoch_row = {
            "epoch": epoch,
            "train_objective_loss": avg_train_loss,
            "val_objective_loss": val_objective_loss,
            "macro_rmse": val_results["macro_rmse"],
            "macro_mae": val_results["macro_mae"],
            "macro_pearson": val_results["macro_pearson"],
            "learning_rate": current_lr,
        }

        for dim_name in target_dims:
            epoch_row[f"{dim_name}_rmse"] = (
                val_results["per_dim_rmse"][dim_name]
            )

            epoch_row[f"{dim_name}_mae"] = (
                val_results["per_dim_mae"][dim_name]
            )

            epoch_row[f"{dim_name}_pearson"] = (
                val_results["per_dim_pearson"][dim_name]
            )

        for group_name, metrics in val_results[
            "group_metrics"
        ].items():
            epoch_row[f"{group_name}_mean_rmse"] = (
                metrics["mean_rmse"]
            )

            epoch_row[f"{group_name}_mean_mae"] = (
                metrics["mean_mae"]
            )

            epoch_row[f"{group_name}_mean_pearson"] = (
                metrics["mean_pearson"]
            )

        epoch_rows.append(epoch_row)

        pd.DataFrame(epoch_rows).to_csv(
            training_log_path,
            index=False,
        )

        # ====================================================
        # Checkpointing and early stopping
        # ====================================================

        if val_objective_loss < best_val_loss:
            best_val_loss = val_objective_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_objective_loss": best_val_loss,
                    "loss_mode": loss_mode,
                    "target_dims": target_dims,
                    "objective_groups": objective_groups,
                },
                checkpoint_path,
            )

            metrics_path = checkpoint_path.with_suffix(
                ".metrics.json"
            )

            with open(metrics_path, "w", encoding="utf-8") as file:
                json.dump(
                    _json_safe({
                        "epoch": epoch,
                        "best_val_objective_loss": best_val_loss,
                        "macro_rmse": val_results["macro_rmse"],
                        "macro_mae": val_results["macro_mae"],
                        "macro_pearson": val_results[
                            "macro_pearson"
                        ],
                        "group_metrics": val_results[
                            "group_metrics"
                        ],
                        "per_dim_rmse": val_results[
                            "per_dim_rmse"
                        ],
                        "per_dim_mae": val_results[
                            "per_dim_mae"
                        ],
                        "per_dim_pearson": val_results[
                            "per_dim_pearson"
                        ],
                        "ranking_metrics": val_results[
                            "ranking_metrics"
                        ],
                        "high_intensity_metrics": val_results[
                            "high_intensity_metrics"
                        ],
                    }),
                    file,
                    indent=2,
                )

            print("New best checkpoint saved.")

        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    return batch_loss_record