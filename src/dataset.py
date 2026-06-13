from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AppraisalDataset(Dataset):
   
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        target_dims: list[str],
        text_column: str = "generated_text",
        max_length: int = 128,
        weights: Optional[dict] = None,
    ):
        self.target_dims = list(target_dims)

        df = pd.read_csv(csv_path)

        required_columns = [text_column] + self.target_dims
        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns in {csv_path}: "
                f"{missing_columns}"
            )

        texts = df[text_column].fillna("").astype(str).tolist()

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        self.input_ids = tokenized["input_ids"]
        self.attention_mask = tokenized["attention_mask"]

        label_array = df[self.target_dims].to_numpy(dtype=np.float32)

        # True where a real annotation exists.
        valid_mask = np.isfinite(label_array)

        # Temporarily replace missing labels with zero to prevent these values from contributing to loss 
        safe_labels = np.nan_to_num(
            label_array,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        self.labels = torch.tensor(
            safe_labels,
            dtype=torch.float32,
        )

        self.valid_mask = torch.tensor(
            valid_mask,
            dtype=torch.bool,
        )

        self.sample_weights = self._build_sample_weights(
            label_array=label_array,
            valid_mask=valid_mask,
            weights=weights,
        )

    def _build_sample_weights(
        self,
        label_array: np.ndarray,
        valid_mask: np.ndarray,
        weights: Optional[dict],
    ) -> torch.Tensor:

        num_examples, num_dims = label_array.shape

        sample_weights = np.ones(
            shape=(num_examples, num_dims),
            dtype=np.float32,
        )

        if weights is None:
            return torch.tensor(
                sample_weights,
                dtype=torch.float32,
            )

        for dim_idx, dim_name in enumerate(self.target_dims):
            if dim_name not in weights:
                raise KeyError(
                    f"No inverse-frequency weights found for "
                    f"dimension: {dim_name}"
                )

            dim_weight_map = {
                int(rating): float(weight)
                for rating, weight in weights[dim_name].items()
            }

            for row_idx in range(num_examples):
                if not valid_mask[row_idx, dim_idx]:
                    sample_weights[row_idx, dim_idx] = 0.0
                    continue

                normalized_value = label_array[row_idx, dim_idx]

                original_rating = int(
                    np.clip(
                        np.rint(normalized_value * 4.0 + 1.0),
                        1,
                        5,
                    )
                )

                sample_weights[row_idx, dim_idx] = (
                    dim_weight_map.get(original_rating, 1.0)
                )

        return torch.tensor(
            sample_weights,
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "weights": self.sample_weights[idx],
            "valid_mask": self.valid_mask[idx],
        }