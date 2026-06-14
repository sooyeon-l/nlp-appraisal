from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.config import (
    DROPOUT_P,
    GROUP_HIDDEN_DIM,
    MAX_LENGTH,
    MODEL_NAME,
    OBJECTIVE_GROUPS,
    SHARED_DIM,
    TARGET_DIMS,
)
from src.covidet_tokenization import (
    encode_sliding_windows,
    encode_truncated,
)
from src.model import build_model


FINAL_RUNS = {
    42: "grouped_sequential_ft_mse_seed42",
    123: "grouped_sequential_ft_mse_seed123",
    456: "grouped_sequential_ft_mse_seed456",
}

TRUNCATION_MODES = [
    "head_128",
    "tail_128",
    "head_tail_128",
]

SLIDING_MODE = "sliding_window_128_stride64"


class CovidETTruncatedDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        mode: str,
        max_length: int,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]
        text = str(row["raw_text"])

        encoded = encode_truncated(
            text=text,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            mode=self.mode,
        )

        full_content_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        retained_content_tokens = min(
            len(full_content_ids),
            self.max_length - 2,
        )

        retained_fraction = (
            retained_content_tokens / len(full_content_ids)
            if full_content_ids
            else 1.0
        )

        return {
            "row_index": index,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded[
                "attention_mask"
            ].squeeze(0),
            "full_content_token_count": len(
                full_content_ids
            ),
            "retained_content_tokens": (
                retained_content_tokens
            ),
            "retained_fraction": retained_fraction,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run zero-shot CovidET inference using the "
            "three final CPM-sequential checkpoints."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            Path("data")
            / "covidet"
            / "covidet_posts_preprocessed.csv"
        ),
    )

    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("outputs")
            / "covidet"
            / "predictions"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--skip-sliding",
        action="store_true",
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


def load_model(
    run_dir: Path,
    device: str,
):
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best_model.pt"

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        run_config = json.load(file)

    model = build_model(
        model_type=run_config["model_type"],
        model_name=run_config.get(
            "model_name",
            MODEL_NAME,
        ),
        target_dims=run_config.get(
            "target_dims",
            TARGET_DIMS,
        ),
        objective_groups=run_config.get(
            "objective_groups",
            OBJECTIVE_GROUPS,
        ),
        shared_dim=run_config.get(
            "shared_dim",
            SHARED_DIM,
        ),
        group_hidden_dim=run_config.get(
            "group_hidden_dim",
            GROUP_HIDDEN_DIM,
        ),
        dropout_p=run_config.get(
            "dropout",
            DROPOUT_P,
        ),
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
    model.eval()

    return model


def base_output_frame(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    keep_columns = [
        column
        for column in [
            "reddit_id",
            "raw_text",
            "character_count",
            "word_count",
            "token_count",
            "length_bin",
            "exceeds_128",
            "retained_fraction_at_128",
        ]
        if column in dataframe.columns
    ]

    return dataframe[keep_columns].copy()


def run_truncated_condition(
    model,
    dataframe: pd.DataFrame,
    tokenizer,
    mode: str,
    max_length: int,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    dataset = CovidETTruncatedDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        mode=mode,
        max_length=max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    predictions = np.empty(
        (len(dataset), len(TARGET_DIMS)),
        dtype=np.float32,
    )

    full_counts = np.empty(
        len(dataset),
        dtype=np.int32,
    )

    retained_counts = np.empty(
        len(dataset),
        dtype=np.int32,
    )

    retained_fractions = np.empty(
        len(dataset),
        dtype=np.float32,
    )

    with torch.no_grad():
        for batch in loader:
            row_indices = batch[
                "row_index"
            ].numpy()

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            batch_predictions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            predictions[row_indices] = (
                batch_predictions
                .detach()
                .cpu()
                .numpy()
            )

            full_counts[row_indices] = batch[
                "full_content_token_count"
            ].numpy()

            retained_counts[row_indices] = batch[
                "retained_content_tokens"
            ].numpy()

            retained_fractions[row_indices] = batch[
                "retained_fraction"
            ].numpy()

    output = base_output_frame(dataframe)

    output["condition"] = mode
    output["full_content_token_count"] = (
        full_counts
    )
    output["retained_content_tokens"] = (
        retained_counts
    )
    output["retained_fraction"] = (
        retained_fractions
    )
    output["window_count"] = 1

    for dim_idx, dimension in enumerate(
        TARGET_DIMS
    ):
        output[
            f"{dimension}_pred"
        ] = predictions[:, dim_idx]

    return output


def run_sliding_condition(
    model,
    dataframe: pd.DataFrame,
    tokenizer,
    max_length: int,
    stride: int,
    device: str,
) -> pd.DataFrame:
    all_predictions = []
    full_counts = []
    retained_counts = []
    retained_fractions = []
    window_counts = []

    with torch.no_grad():
        for row_idx, row in dataframe.iterrows():
            text = str(row["raw_text"])

            full_content_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
            )

            windows = encode_sliding_windows(
                text=text,
                tokenizer=tokenizer,
                max_length=max_length,
                stride=stride,
            )

            window_predictions = []

            for window in windows:
                input_ids = window[
                    "input_ids"
                ].to(device)

                attention_mask = window[
                    "attention_mask"
                ].to(device)

                prediction = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                window_predictions.append(
                    prediction
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )

            mean_prediction = np.mean(
                np.stack(
                    window_predictions,
                    axis=0,
                ),
                axis=0,
            )

            all_predictions.append(
                mean_prediction
            )

            full_count = len(full_content_ids)

            covered_positions = set()

            for window in windows:
                covered_positions.update(
                    range(
                        int(window["content_start"]),
                        int(window["content_end"]),
                    )
                )

            retained_count = len(
                covered_positions
            )

            retained_fraction = (
                retained_count / full_count
                if full_count
                else 1.0
            )

            full_counts.append(full_count)
            retained_counts.append(
                retained_count
            )
            retained_fractions.append(
                retained_fraction
            )
            window_counts.append(
                len(windows)
            )

            if (
                (row_idx + 1) % 25 == 0
                or row_idx + 1 == len(dataframe)
            ):
                print(
                    f"Sliding inference: "
                    f"{row_idx + 1}/{len(dataframe)}",
                    flush=True,
                )

    predictions = np.stack(
        all_predictions,
        axis=0,
    )

    output = base_output_frame(dataframe)

    output["condition"] = SLIDING_MODE
    output["full_content_token_count"] = (
        full_counts
    )
    output["retained_content_tokens"] = (
        retained_counts
    )
    output["retained_fraction"] = (
        retained_fractions
    )
    output["window_count"] = (
        window_counts
    )

    for dim_idx, dimension in enumerate(
        TARGET_DIMS
    ):
        output[
            f"{dimension}_pred"
        ] = predictions[:, dim_idx]

    return output


def save_ensemble(
    seed_paths: list[Path],
    output_path: Path,
) -> None:
    dataframes = [
        pd.read_csv(path)
        for path in seed_paths
    ]

    reference = dataframes[0].copy()

    prediction_columns = [
        f"{dimension}_pred"
        for dimension in TARGET_DIMS
    ]

    for dataframe in dataframes[1:]:
        if not reference[
            "reddit_id"
        ].equals(dataframe["reddit_id"]):
            raise ValueError(
                "Reddit ID order differs across seeds."
            )

    for column in prediction_columns:
        reference[column] = np.mean(
            np.column_stack([
                dataframe[column].to_numpy(
                    dtype=float
                )
                for dataframe in dataframes
            ]),
            axis=1,
        )

    reference["ensemble_n_seeds"] = len(
        dataframes
    )

    reference.to_csv(
        output_path,
        index=False,
    )


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = choose_device(args.device)

    dataframe = pd.read_csv(args.input)

    if "raw_text" not in dataframe.columns:
        raise ValueError(
            "Input must contain a raw_text column."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Input:", args.input)
    print("Posts:", len(dataframe))
    print("Device:", device)
    print("Output:", args.output_dir)
    print()

    conditions = list(
        TRUNCATION_MODES
    )

    if not args.skip_sliding:
        conditions.append(
            SLIDING_MODE
        )

    condition_seed_paths = {
        condition: []
        for condition in conditions
    }

    for seed, run_name in FINAL_RUNS.items():
        run_dir = (
            args.runs_root
            / run_name
        )

        print("=" * 80)
        print(
            f"Loading seed {seed}: {run_dir}"
        )

        model = load_model(
            run_dir=run_dir,
            device=device,
        )

        for mode in TRUNCATION_MODES:
            print(
                f"Seed {seed}: {mode}",
                flush=True,
            )

            output = run_truncated_condition(
                model=model,
                dataframe=dataframe,
                tokenizer=tokenizer,
                mode=mode,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=device,
            )

            output_path = (
                args.output_dir
                / f"seed{seed}_{mode}.csv"
            )

            output.to_csv(
                output_path,
                index=False,
            )

            condition_seed_paths[
                mode
            ].append(output_path)

            print("Saved:", output_path)

        if not args.skip_sliding:
            print(
                f"Seed {seed}: {SLIDING_MODE}",
                flush=True,
            )

            output = run_sliding_condition(
                model=model,
                dataframe=dataframe,
                tokenizer=tokenizer,
                max_length=args.max_length,
                stride=args.stride,
                device=device,
            )

            output_path = (
                args.output_dir
                / (
                    f"seed{seed}_"
                    f"{SLIDING_MODE}.csv"
                )
            )

            output.to_csv(
                output_path,
                index=False,
            )

            condition_seed_paths[
                SLIDING_MODE
            ].append(output_path)

            print("Saved:", output_path)

        del model

        if device == "cuda":
            torch.cuda.empty_cache()

    print()
    print("=" * 80)
    print("Creating three-seed ensembles")
    print("=" * 80)

    for condition, seed_paths in (
        condition_seed_paths.items()
    ):
        ensemble_path = (
            args.output_dir
            / f"ensemble_{condition}.csv"
        )

        save_ensemble(
            seed_paths=seed_paths,
            output_path=ensemble_path,
        )

        print("Saved:", ensemble_path)


if __name__ == "__main__":
    main()
