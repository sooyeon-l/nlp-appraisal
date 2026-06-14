from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

bootstrap_script = (
    PROJECT_ROOT
    / "analysis"
    / "paired_bootstrap_validation.py"
)

spec = importlib.util.spec_from_file_location(
    "bootstrap_module",
    bootstrap_script,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


rows = []

for model_a_name, model_b_name in module.COMPARISONS:
    model_a_tag = module.ARCHITECTURES[
        model_a_name
    ]

    model_b_tag = module.ARCHITECTURES[
        model_b_name
    ]

    for seed in module.SEEDS:
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

        for metric_name in module.METRICS:
            result = module.paired_bootstrap(
                predictions_a=predictions_a,
                predictions_b=predictions_b,
                labels=labels_a,
                valid_mask=mask_a,
                metric_name=metric_name,
                n_bootstraps=2000,
                random_seed=2026 + seed,
            )

            result.update({
                "Seed": seed,
                "Model A": model_a_name,
                "Model B": model_b_name,
            })

            rows.append(result)


seed_results_df = pd.DataFrame(rows)

output_path = (
    module.OUTPUT_DIR
    / "paired_bootstrap_validation_by_seed.csv"
)

seed_results_df.to_csv(
    output_path,
    index=False,
)

print("Saved:", output_path)
print(seed_results_df.to_string(index=False))