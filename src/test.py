from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import (
    BATCH_SIZE,
    DROPOUT_P,
    GROUP_HIDDEN_DIM,
    MAX_LENGTH,
    MODEL_NAME,
    OBJECTIVE_GROUPS,
    SAVE_PATH,
    SHARED_DIM,
    TARGET_DIMS,
    TEXT_COLUMN,
)
from src.dataset import AppraisalDataset
from src.eval import evaluate_model
from src.model import build_model


def json_safe(value):
    if isinstance(value, dict):
        return {
            key: json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run folder under SAVE_PATH/runs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    save_path = Path(SAVE_PATH)
    run_dir = save_path / "runs" / args.run

    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best_model.pt"

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    with open(config_path, "r", encoding="utf-8") as file:
        run_config = json.load(file)

    model_type = run_config["model_type"]
    loss_mode = run_config["loss"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Weights are needed only for weighted evaluation objectives.
    weights_json = None
    weights_path = save_path / "dim_weights.json"

    if loss_mode in {
        "weighted_mse",
        "weighted_group_balanced_mse",
    }:
        with open(weights_path, "r", encoding="utf-8") as file:
            weights_json = json.load(file)

    test_dataset = AppraisalDataset(
        csv_path=str(save_path / "test.csv"),
        tokenizer=tokenizer,
        target_dims=TARGET_DIMS,
        text_column=TEXT_COLUMN,
        max_length=MAX_LENGTH,
        weights=weights_json,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        model_type=model_type,
        model_name=MODEL_NAME,
        target_dims=TARGET_DIMS,
        objective_groups=OBJECTIVE_GROUPS,
        shared_dim=SHARED_DIM,
        group_hidden_dim=GROUP_HIDDEN_DIM,
        dropout_p=DROPOUT_P,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        target_dims=TARGET_DIMS,
        objective_groups=OBJECTIVE_GROUPS,
        loss_mode=loss_mode,
        device=device,
    )

    # -----------------------------------------------
    # Save metrics
    # -----------------------------------------------

    metrics = {
        key: value
        for key, value in results.items()
        if key not in {
            "all_predictions",
            "all_labels",
            "all_masks",
        }
    }

    with open(
        run_dir / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(metrics),
            file,
            indent=2,
        )

    # -----------------------------------------------
    # Save predictions
    # -----------------------------------------------

    test_df = pd.read_csv(save_path / "test.csv")

    predictions = results["all_predictions"]
    labels = results["all_labels"]
    masks = results["all_masks"]

    output_df = pd.DataFrame()

    if TEXT_COLUMN in test_df.columns:
        output_df[TEXT_COLUMN] = test_df[TEXT_COLUMN]

    for dim_idx, dim_name in enumerate(TARGET_DIMS):
        output_df[f"{dim_name}_true"] = np.where(
            masks[:, dim_idx],
            labels[:, dim_idx],
            np.nan,
        )

        output_df[f"{dim_name}_pred"] = predictions[:, dim_idx]

    output_df.to_csv(
        run_dir / "test_predictions.csv",
        index=False,
    )

    print(f"Run: {args.run}")
    print(f"Test macro RMSE: {results['macro_rmse']:.4f}")
    print(f"Test macro MAE: {results['macro_mae']:.4f}")
    print(
        f"Test macro Pearson: "
        f"{results['macro_pearson']:.4f}"
    )


if __name__ == "__main__":
    main()