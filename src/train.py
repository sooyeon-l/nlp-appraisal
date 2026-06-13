from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import (
    BATCH_SIZE,
    DROPOUT_P,
    EARLY_STOPPING_PATIENCE,
    GRAD_CLIP,
    GROUP_HIDDEN_DIM,
    LOSS_TYPES,
    LR_BASE_MODEL,
    LR_HEAD,
    LR_HEAD_FINETUNE,
    MAX_LENGTH,
    MODEL_NAME,
    MODEL_TYPES,
    N_EPOCHS,
    OBJECTIVE_GROUPS,
    RANDOM_SEED,
    SAVE_PATH,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SHARED_DIM,
    TARGET_DIMS,
    TEXT_COLUMN,
    WEIGHT_DECAY,
)
from src.dataset import AppraisalDataset
from src.model import (
    build_model,
    freeze_encoder,
    print_trainable_parameters,
    unfreeze_encoder,
)
from src.trainer import train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Unique run name.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=MODEL_TYPES,
    )

    parser.add_argument(
        "--loss",
        type=str,
        required=True,
        choices=LOSS_TYPES,
    )

    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["head", "finetune"],
        help=(
            "'head' freezes RoBERTa. "
            "'finetune' unfreezes RoBERTa."
        ),
    )

    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint used to initialize fine-tuning, usually "
            "the best checkpoint from the corresponding head-only run."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    save_path = Path(SAVE_PATH)
    save_path.mkdir(parents=True, exist_ok=True)

    run_dir = save_path / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "best_model.pt"
    training_log_path = run_dir / "training_log.csv"
    batch_loss_path = run_dir / "batch_losses.csv"
    config_path = run_dir / "config.json"

    weights_path = save_path / "dim_weights.json"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Could not find inverse-frequency weights at "
            f"{weights_path}"
        )

    with open(weights_path, "r", encoding="utf-8") as file:
        weights_json = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = AppraisalDataset(
        csv_path=str(save_path / "train.csv"),
        tokenizer=tokenizer,
        target_dims=TARGET_DIMS,
        text_column=TEXT_COLUMN,
        max_length=MAX_LENGTH,
        weights=weights_json,
    )

    val_dataset = AppraisalDataset(
        csv_path=str(save_path / "val.csv"),
        tokenizer=tokenizer,
        target_dims=TARGET_DIMS,
        text_column=TEXT_COLUMN,
        max_length=MAX_LENGTH,
        weights=weights_json,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        model_type=args.model_type,
        model_name=MODEL_NAME,
        target_dims=TARGET_DIMS,
        objective_groups=OBJECTIVE_GROUPS,
        shared_dim=SHARED_DIM,
        group_hidden_dim=GROUP_HIDDEN_DIM,
        dropout_p=DROPOUT_P,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Architecture: {args.model_type}")
    print(f"Loss: {args.loss}")
    print(f"Stage: {args.stage}")

    # ========================================================
    # Optional initialization from head-only checkpoint
    # ========================================================

    if args.init_checkpoint is not None:
        init_checkpoint = Path(args.init_checkpoint)

        if not init_checkpoint.exists():
            raise FileNotFoundError(
                f"Initialization checkpoint not found: "
                f"{init_checkpoint}"
            )

        checkpoint = torch.load(
            init_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            f"Initialized model from: {init_checkpoint}"
        )

    model.to(device)

    # ========================================================
    # Freeze or unfreeze transformer
    # ========================================================

    if args.stage == "head":
        freeze_encoder(model)

        optimizer = AdamW(
            model.head_parameters(),
            lr=LR_HEAD,
            weight_decay=WEIGHT_DECAY,
        )

    else:
        unfreeze_encoder(model)

        optimizer = AdamW(
            [
                {
                    "params": model.base_model.parameters(),
                    "lr": LR_BASE_MODEL,
                    "weight_decay": WEIGHT_DECAY,
                },
                {
                    "params": list(model.head_parameters()),
                    "lr": LR_HEAD_FINETUNE,
                    "weight_decay": WEIGHT_DECAY,
                },
            ]
        )

    print_trainable_parameters(model)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )

    config_snapshot = {
        "run": args.run,
        "model_type": args.model_type,
        "loss": args.loss,
        "stage": args.stage,
        "init_checkpoint": args.init_checkpoint,
        "model_name": MODEL_NAME,
        "num_targets": len(TARGET_DIMS),
        "target_dims": TARGET_DIMS,
        "objective_groups": OBJECTIVE_GROUPS,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "dropout": DROPOUT_P,
        "shared_dim": SHARED_DIM,
        "group_hidden_dim": GROUP_HIDDEN_DIM,
        "lr_head": LR_HEAD,
        "lr_base_model": LR_BASE_MODEL,
        "lr_head_finetune": LR_HEAD_FINETUNE,
        "weight_decay": WEIGHT_DECAY,
        "seed": args.seed,
    }

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(
            config_snapshot,
            file,
            indent=2,
        )

    batch_losses = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=N_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        checkpoint_path=checkpoint_path,
        training_log_path=training_log_path,
        target_dims=TARGET_DIMS,
        objective_groups=OBJECTIVE_GROUPS,
        loss_mode=args.loss,
        device=device,
        grad_clip=GRAD_CLIP,
    )

    pd.DataFrame({
        "batch_loss": batch_losses
    }).to_csv(
        batch_loss_path,
        index=False,
    )


if __name__ == "__main__":
    main()