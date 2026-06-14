from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.appraisal_emotion_mapping import (
    appraisal_to_emotion_scores,
    rank_scores,
)
from src.emotion_model import GoEmotionsClassifier
from src.integrated_emotion import compare_branches
from src.model import build_model


FINAL_RUNS = {
    42: "grouped_sequential_ft_mse_seed42",
    123: "grouped_sequential_ft_mse_seed123",
    456: "grouped_sequential_ft_mse_seed456",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare GoEmotions-only, appraisal-only, and integrated "
            "emotion candidates on a fixed journal-entry development set."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "emotion_branch" / "dev_entries.csv",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "emotion_branch",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--goemotions-model",
        default="SamLowe/roberta-base-go_emotions",
    )
    parser.add_argument(
        "--goemotions-threshold",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--emotion-weight",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--appraisal-weight",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--appraisal-max-length",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--appraisal-stride",
        type=int,
        default=64,
    )
    return parser.parse_args()


def choose_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def appraisal_windows(
    text: str,
    tokenizer,
    max_length: int,
    stride: int,
) -> list[dict[str, torch.Tensor]]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    windows = []

    for index in range(encoded["input_ids"].shape[0]):
        windows.append({
            "input_ids": encoded["input_ids"][index:index + 1],
            "attention_mask": encoded["attention_mask"][index:index + 1],
        })

    return windows


def load_appraisal_model(
    run_dir: Path,
    device: str,
):
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best_model.pt"

    for path in [config_path, checkpoint_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    run_config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    model = build_model(
        model_type=run_config["model_type"],
        model_name=run_config["model_name"],
        target_dims=run_config["target_dims"],
        objective_groups=run_config["objective_groups"],
        shared_dim=run_config["shared_dim"],
        group_hidden_dim=run_config["group_hidden_dim"],
        dropout_p=run_config["dropout"],
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, run_config


@torch.inference_mode()
def predict_appraisals_for_texts(
    model,
    tokenizer,
    texts: list[str],
    target_dims: list[str],
    max_length: int,
    stride: int,
    device: str,
) -> tuple[np.ndarray, list[int]]:
    all_predictions = []
    window_counts = []

    for text_index, text in enumerate(texts):
        windows = appraisal_windows(
            text=text,
            tokenizer=tokenizer,
            max_length=max_length,
            stride=stride,
        )

        window_predictions = []

        for window in windows:
            prediction = model(
                input_ids=window["input_ids"].to(device),
                attention_mask=window["attention_mask"].to(device),
            )
            window_predictions.append(
                prediction.squeeze(0).detach().cpu().numpy()
            )

        # Mean aggregation matches the earlier CovidET sliding analysis.
        entry_prediction = np.mean(
            np.stack(window_predictions, axis=0),
            axis=0,
        )
        all_predictions.append(entry_prediction)
        window_counts.append(len(windows))

        print(
            f"  Appraisal entry {text_index + 1}/{len(texts)}",
            flush=True,
        )

    return np.stack(all_predictions, axis=0), window_counts


def format_labels(items: list[dict[str, float]], n: int) -> str:
    return "; ".join(
        f"{item['label']}={item['score']:.3f}"
        for item in items[:n]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    device = choose_device(args.device)
    dev = pd.read_csv(args.input)

    required = {"entry_id", "text"}
    missing = required - set(dev.columns)
    if missing:
        raise ValueError(
            "Input is missing columns: " + ", ".join(sorted(missing))
        )

    texts = dev["text"].fillna("").astype(str).tolist()

    print("Device:", device)
    print("Entries:", len(texts))
    print()

    print("Loading GoEmotions model...")
    emotion_model = GoEmotionsClassifier(
        model_name=args.goemotions_model,
        device=device,
        threshold=args.goemotions_threshold,
    )

    goemotions_results = []
    for index, text in enumerate(texts):
        result = emotion_model.predict(
            text=text,
            top_k=args.top_k,
        )
        goemotions_results.append(result)
        print(
            f"GoEmotions entry {index + 1}/{len(texts)}",
            flush=True,
        )

    # Free GPU before loading the much larger appraisal checkpoints.
    del emotion_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    seed_predictions = []
    target_dims = None
    appraisal_window_counts = None

    for seed, run_name in FINAL_RUNS.items():
        print()
        print("=" * 72)
        print(f"Loading appraisal seed {seed}")

        model, run_config = load_appraisal_model(
            args.runs_root / run_name,
            device=device,
        )

        if target_dims is None:
            target_dims = run_config["target_dims"]
            tokenizer = AutoTokenizer.from_pretrained(
                run_config["model_name"]
            )
        elif target_dims != run_config["target_dims"]:
            raise ValueError("Target dimension order differs across seeds.")

        predictions, counts = predict_appraisals_for_texts(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            target_dims=target_dims,
            max_length=args.appraisal_max_length,
            stride=args.appraisal_stride,
            device=device,
        )
        seed_predictions.append(predictions)

        if appraisal_window_counts is None:
            appraisal_window_counts = counts
        elif appraisal_window_counts != counts:
            raise ValueError("Window counts differ across seeds.")

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    appraisal_ensemble = np.mean(
        np.stack(seed_predictions, axis=0),
        axis=0,
    )

    output_rows = []

    for index, row in dev.iterrows():
        appraisal_profile = {
            dimension: float(appraisal_ensemble[index, dim_index])
            for dim_index, dimension in enumerate(target_dims)
        }

        appraisal_emotions = appraisal_to_emotion_scores(
            appraisal_profile
        )

        emotion_scores = goemotions_results[index].scores

        comparison = compare_branches(
            emotion_scores=emotion_scores,
            appraisal_scores=appraisal_emotions,
            top_k=args.top_k,
            emotion_weight=args.emotion_weight,
            appraisal_weight=args.appraisal_weight,
        )

        emotion_ranked = comparison["emotion_only"]
        appraisal_ranked = comparison["appraisal_only"]
        integrated_ranked = comparison["integrated"]

        output_rows.append({
            **row.to_dict(),
            "goemotions_window_count": (
                goemotions_results[index].window_count
            ),
            "appraisal_window_count": appraisal_window_counts[index],
            "emotion_only_top1": format_labels(emotion_ranked, 1),
            "emotion_only_top3": format_labels(emotion_ranked, 3),
            "emotion_only_top5": format_labels(emotion_ranked, 5),
            "appraisal_only_top1": format_labels(appraisal_ranked, 1),
            "appraisal_only_top3": format_labels(appraisal_ranked, 3),
            "appraisal_only_top5": format_labels(appraisal_ranked, 5),
            "integrated_top1": format_labels(integrated_ranked, 1),
            "integrated_top3": format_labels(integrated_ranked, 3),
            "integrated_top5": format_labels(integrated_ranked, 5),
            "goemotions_selected_at_threshold": json.dumps(
                goemotions_results[index].selected,
                ensure_ascii=False,
            ),
            "goemotions_scores_json": json.dumps(
                emotion_scores,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "appraisal_profile_json": json.dumps(
                appraisal_profile,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "appraisal_emotion_scores_json": json.dumps(
                appraisal_emotions,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "integrated_scores_json": json.dumps(
                comparison["integrated_scores"],
                ensure_ascii=False,
                sort_keys=True,
            ),
            # Blank columns for development review.
            "emotion_only_relevance_1to5": "",
            "appraisal_only_relevance_1to5": "",
            "integrated_relevance_1to5": "",
            "emotion_only_specificity_1to5": "",
            "appraisal_only_specificity_1to5": "",
            "integrated_specificity_1to5": "",
            "best_branch": "",
            "missing_emotions": "",
            "irrelevant_emotions": "",
            "review_notes": "",
        })

    output = pd.DataFrame(output_rows)

    output_path = args.output_dir / "emotion_branch_comparison.csv"
    output.to_csv(output_path, index=False)

    metadata = {
        "goemotions_model": args.goemotions_model,
        "goemotions_threshold": args.goemotions_threshold,
        "appraisal_runs": FINAL_RUNS,
        "appraisal_aggregation": (
            "mean across sliding windows, then mean across three seeds"
        ),
        "emotion_weight": args.emotion_weight,
        "appraisal_weight": args.appraisal_weight,
        "top_k": args.top_k,
        "n_entries": len(output),
    }
    metadata_path = args.output_dir / "emotion_branch_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("Saved:", output_path)
    print("Saved:", metadata_path)
    print()
    print(
        output[
            [
                "entry_id",
                "emotion_only_top3",
                "appraisal_only_top3",
                "integrated_top3",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
