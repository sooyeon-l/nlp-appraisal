from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_GOEMOTIONS_MODEL = "SamLowe/roberta-base-go_emotions"


@dataclass
class EmotionPrediction:
    scores: dict[str, float]
    selected: list[dict[str, float]]
    top_k: list[dict[str, float]]
    window_count: int


class GoEmotionsClassifier:
    """
    Multi-label GoEmotions inference with overflow-window support.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GOEMOTIONS_MODEL,
        device: str = "auto",
        threshold: float = 0.30,
        max_length: int = 512,
        stride: int = 256,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.max_length = max_length
        self.stride = stride
        self.device = self._choose_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.model.to(self.device)
        self.model.eval()

        self.id2label = {
            int(index): label
            for index, label in self.model.config.id2label.items()
        }

    @staticmethod
    def _choose_device(requested: str) -> str:
        if requested == "cpu":
            return "cpu"
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable.")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _encode_windows(self, text: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    @torch.inference_mode()
    def predict(
        self,
        text: str,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> EmotionPrediction:
        threshold = self.threshold if threshold is None else threshold
        windows = self._encode_windows(text)

        input_ids = windows["input_ids"].to(self.device)
        attention_mask = windows["attention_mask"].to(self.device)

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

        probabilities = torch.sigmoid(logits).detach().cpu().numpy()

        # Label-wise max pooling preserves an emotion that appears strongly
        # in any one window of a longer journal entry.
        pooled = np.max(probabilities, axis=0)

        scores = {
            self.id2label[index]: float(pooled[index])
            for index in range(len(pooled))
        }

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        selected = [
            {"label": label, "score": score}
            for label, score in ranked
            if score >= threshold
        ]

        if not selected and ranked:
            selected = [
                {"label": ranked[0][0], "score": ranked[0][1]}
            ]

        top_items = [
            {"label": label, "score": score}
            for label, score in ranked[:top_k]
        ]

        return EmotionPrediction(
            scores=scores,
            selected=selected,
            top_k=top_items,
            window_count=int(input_ids.shape[0]),
        )
