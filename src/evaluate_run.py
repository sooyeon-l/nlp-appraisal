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
    SHARED_DIM,
    TARGET_DIMS,
    TEXT_COLUMN,
)
from src.dataset import AppraisalDataset
from src.eval import evaluate_model
from src.model import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
DATA_ROOT = PROJECT_ROOT / "data"


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
        required=True,
        help="Run folder inside the local runs directory.",
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    return parser.parse_args()


def choose_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()

    run_dir = RUNS_ROOT / args.run
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best_model.pt"
    split_path = DATA_ROOT / f"{args.split}.csv"

    for path in [
        run_dir,
        config_path,
        checkpoint_path,
        split_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    with open(config_path, "r", encoding="utf-8") as file:
        run_config = json.load(file)

    model_type = run_config["model_type"]
    loss_mode = run_config["loss"]

    device = choose_device(args.device)

    print(f"Run: {args.run}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    weights_json = None
    weights_path = DATA_ROOT / "dim_weights.json"

    if loss_mode in {
        "weighted_mse",
        "weighted_group_balanced_mse",
    }:
        if not weights_path.exists():
            raise FileNotFoundError(weights_path)

        with open(
            weights_path,
            "r",
            encoding="utf-8",
        ) as file:
            weights_json = json.load(file)

    dataset = AppraisalDataset(
        csv_path=str(split_path),
        tokenizer=tokenizer,
        target_dims=TARGET_DIMS,
        text_column=TEXT_COLUMN,
        max_length=MAX_LENGTH,
        weights=weights_json,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
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

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    results = evaluate_model(
        model=model,
        dataloader=loader,
        target_dims=TARGET_DIMS,
        objective_groups=OBJECTIVE_GROUPS,
        loss_mode=loss_mode,
        device=device,
    )

    metrics = {
        key: value
        for key, value in results.items()
        if key not in {
            "all_predictions",
            "all_labels",
            "all_masks",
        }
    }

    metrics_output_path = (
        run_dir
        / f"{args.split}_metrics_recomputed.json"
    )

    with open(
        metrics_output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(metrics),
            file,
            indent=2,
        )

    split_df = pd.read_csv(split_path)

    predictions = results["all_predictions"]
    labels = results["all_labels"]
    masks = results["all_masks"]

    output_df = pd.DataFrame()

    if TEXT_COLUMN in split_df.columns:
        output_df[TEXT_COLUMN] = split_df[TEXT_COLUMN]

    for dim_idx, dim_name in enumerate(TARGET_DIMS):
        output_df[f"{dim_name}_true"] = np.where(
            masks[:, dim_idx],
            labels[:, dim_idx],
            np.nan,
        )

        output_df[f"{dim_name}_pred"] = (
            predictions[:, dim_idx]
        )

    predictions_output_path = (
        run_dir
        / f"{args.split}_predictions_recomputed.csv"
    )

    output_df.to_csv(
        predictions_output_path,
        index=False,
    )

    print(
        f"Macro RMSE: "
        f"{results['macro_rmse']:.4f}"
    )
    print(
        f"Macro MAE: "
        f"{results['macro_mae']:.4f}"
    )
    print(
        f"Macro Pearson: "
        f"{results['macro_pearson']:.4f}"
    )

    ranking = results["ranking_metrics"]

    print(
        f"Within-entry Spearman: "
        f"{ranking['mean_within_entry_spearman']:.4f}"
    )
    print(
        f"Top-3 overlap: "
        f"{ranking['top3_overlap']:.4f}"
    )
    print(
        f"Saved metrics: {metrics_output_path}"
    )
    print(
        f"Saved predictions: "
        f"{predictions_output_path}"
    )


if __name__ == "__main__":
    main()