from __future__ import annotations

from typing import Literal

import torch
from transformers import PreTrainedTokenizerBase


TruncationMode = Literal[
    "head_128",
    "tail_128",
    "head_tail_128",
]


def encode_truncated(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    mode: TruncationMode = "head_128",
) -> dict[str, torch.Tensor]:
    content_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    content_limit = max_length - 2

    if mode == "head_128":
        selected = content_ids[
            :content_limit
        ]

    elif mode == "tail_128":
        selected = content_ids[
            -content_limit:
        ]

    elif mode == "head_tail_128":
        head_size = (
            content_limit // 2
        )

        tail_size = (
            content_limit - head_size
        )

        if len(content_ids) <= content_limit:
            selected = content_ids
        else:
            selected = (
                content_ids[:head_size]
                + content_ids[-tail_size:]
            )

    else:
        raise ValueError(
            f"Unknown truncation mode: {mode}"
        )

    prepared = tokenizer.prepare_for_model(
        selected,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )

    return {
        "input_ids": prepared[
            "input_ids"
        ],
        "attention_mask": prepared[
            "attention_mask"
        ],
    }


def encode_sliding_windows(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    stride: int = 64,
) -> list[dict[str, torch.Tensor]]:
    content_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    content_limit = max_length - 2

    if len(content_ids) <= content_limit:
        starts = [0]
    else:
        starts = list(
            range(
                0,
                len(content_ids),
                stride,
            )
        )

        final_start = max(
            0,
            len(content_ids)
            - content_limit,
        )

        if final_start not in starts:
            starts.append(final_start)

    windows = []
    seen = set()

    for start in starts:
        end = (
            start + content_limit
        )

        selected = content_ids[
            start:end
        ]

        key = tuple(selected)

        if key in seen:
            continue

        seen.add(key)

        prepared = (
            tokenizer.prepare_for_model(
                selected,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
        )

        windows.append({
            "input_ids": prepared[
                "input_ids"
            ],
            "attention_mask": prepared[
                "attention_mask"
            ],
            "content_start": start,
            "content_end": min(
                end,
                len(content_ids),
            ),
        })

    return windows
