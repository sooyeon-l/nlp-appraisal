from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


DEFAULT_N_BOOTSTRAPS = 2000
DEFAULT_RANDOM_SEED = 2026


def load_bootstrap_module():
    bootstrap_script = (
        Path(__file__).resolve().parent
        / "paired_bootstrap_validation.py"
    )

    if not bootstrap_script.exists():
        raise FileNotFoundError(
            f"Could not find:\n{bootstrap_script}"
        )

    spec = importlib.util.spec_from_file_location(
        "bootstrap_module",
        bootstrap_script,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import {bootstrap_script}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def run_by_seed_analysis(
    n_bootstraps: int,
    random_seed: int,
    show_progress: bool,
) -> pd.DataFrame:
    module = load_bootstrap_module()

    rows = []

    total_jobs = (
        len(module.COMPARISONS)
        * len(module.SEEDS)
        * len(module.METRICS)
    )
    current_job = 0

    for comparison_idx, (
        model_a_name,
        model_b_name,
    ) in enumerate(module.COMPARISONS):
        model_a_tag = module.ARCHITECTURES[
            model_a_name
        ]

        model_b_tag = module.ARCHITECTURES[
            model_b_name
        ]

        print()
        print("=" * 80)
        print(f"{model_a_name} vs {model_b_name}")
        print("=" * 80)

        for seed_idx, seed in enumerate(module.SEEDS):
            predictions_a, labels_a, mask_a = (
                module.load_prediction_arrays(
                    architecture_tag=model_a_tag,
                    seed=seed,
                )
            )

            predictions_b, labels_b, mask_b = (
                module.load_prediction_arrays(
                    architecture_tag=model_b_tag,
                    seed=seed,
                )
            )

            if not module.np.array_equal(
                mask_a,
                mask_b,
            ):
                raise ValueError(
                    f"Mask mismatch at seed {seed}"
                )

            if not module.np.allclose(
                labels_a[mask_a],
                labels_b[mask_b],
                atol=1e-8,
            ):
                raise ValueError(
                    f"Label mismatch at seed {seed}"
                )

            print(f"\nSeed {seed}", flush=True)

            for metric_idx, metric_name in enumerate(
                module.METRICS
            ):
                current_job += 1

                job_seed = (
                    random_seed
                    + comparison_idx * 100_000
                    + seed_idx * 10_000
                    + metric_idx
                )

                result = module.paired_bootstrap(
                    predictions_a=predictions_a,
                    predictions_b=predictions_b,
                    labels=labels_a,
                    valid_mask=mask_a,
                    metric_name=metric_name,
                    n_bootstraps=n_bootstraps,
                    random_seed=job_seed,
                    show_progress=show_progress,
                    progress_desc=(
                        f"[{current_job}/{total_jobs}] "
                        f"{model_a_name} vs "
                        f"{model_b_name}, "
                        f"seed {seed}: {metric_name}"
                    ),
                )

                result.update({
                    "Seed": seed,
                    "Model A": model_a_name,
                    "Model B": model_b_name,
                })

                rows.append(result)

                print(
                    f"{metric_name:25} "
                    f"diff="
                    f"{result['Difference Favoring A']:+.6f} "
                    f"CI=[{result['95% CI Low']:+.6f}, "
                    f"{result['95% CI High']:+.6f}] "
                    f"P(A better)="
                    f"{result['Probability A Better']:.3f}",
                    flush=True,
                )

    seed_results_df = pd.DataFrame(rows)

    column_order = [
        "Model A",
        "Model B",
        "Seed",
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

    seed_results_df = seed_results_df[column_order]

    output_path = (
        module.OUTPUT_DIR
        / "paired_bootstrap_validation_by_seed.csv"
    )

    seed_results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Saved:", output_path)
    print()
    print(seed_results_df.to_string(index=False))

    return seed_results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired bootstrap validation separately "
            "for each training seed."
        )
    )

    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=DEFAULT_N_BOOTSTRAPS,
        help="Number of paired bootstrap resamples.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = load_bootstrap_module()

    print(f"DATA_ROOT:    {module.DATA_ROOT}")
    print(f"RUNS_ROOT:    {module.RUNS_ROOT}")
    print(f"OUTPUT_DIR:   {module.OUTPUT_DIR}")
    print(f"Bootstraps:   {args.n_bootstraps}")
    print()

    run_by_seed_analysis(
        n_bootstraps=args.n_bootstraps,
        random_seed=args.random_seed,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
