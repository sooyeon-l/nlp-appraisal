from __future__ import annotations

from dataclasses import dataclass

from src.appraisal_emotion_mapping import rank_scores


@dataclass
class FusionResult:
    emotion_only: list[dict[str, float]]
    appraisal_only: list[dict[str, float]]
    integrated: list[dict[str, float]]
    integrated_scores: dict[str, float]
    candidate_labels: list[str]
    added_by_appraisal: str | None


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def candidate_gated_fusion(
    emotion_scores: dict[str, float],
    appraisal_scores: dict[str, float],
    *,
    top_k: int = 5,
    candidate_pool_size: int = 8,
    emotion_weight: float = 0.80,
    appraisal_weight: float = 0.20,
    min_emotion_candidate_score: float = 0.05,
    allow_appraisal_addition: bool = True,
    appraisal_add_threshold: float = 0.72,
    max_added_candidates: int = 1,
    neutral_protection_threshold: float = 0.55,
) -> FusionResult:
    if candidate_pool_size < top_k:
        candidate_pool_size = top_k

    if emotion_weight < 0 or appraisal_weight < 0:
        raise ValueError("Fusion weights must be non-negative.")

    total = emotion_weight + appraisal_weight
    if total <= 0:
        raise ValueError("At least one fusion weight must be positive.")

    emotion_weight /= total
    appraisal_weight /= total

    emotion_ranked = sorted(
        emotion_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    appraisal_ranked = sorted(
        appraisal_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    candidate_labels = [
        label
        for label, score in emotion_ranked[:candidate_pool_size]
        if float(score) >= min_emotion_candidate_score
    ]

    if not candidate_labels and emotion_ranked:
        candidate_labels = [emotion_ranked[0][0]]

    added_by_appraisal = None

    if allow_appraisal_addition and max_added_candidates > 0:
        strong_neutral = (
            float(emotion_scores.get("neutral", 0.0))
            >= neutral_protection_threshold
        )
        additions = 0

        for label, appraisal_score in appraisal_ranked:
            if additions >= max_added_candidates:
                break
            if label in candidate_labels:
                continue
            if float(appraisal_score) < appraisal_add_threshold:
                break
            if strong_neutral and float(emotion_scores.get(label, 0.0)) < 0.10:
                continue

            candidate_labels.append(label)
            added_by_appraisal = label
            additions += 1

    integrated_scores = {}

    for label in candidate_labels:
        emotion_score = _clip(emotion_scores.get(label, 0.0))
        appraisal_score = _clip(appraisal_scores.get(label, 0.0))

        fused = (
            emotion_weight * emotion_score
            + appraisal_weight * appraisal_score
        )

        if label == "neutral":
            fused = emotion_score

        integrated_scores[label] = _clip(fused)

    neutral_score = float(emotion_scores.get("neutral", 0.0))
    if (
        neutral_score >= neutral_protection_threshold
        and "neutral" not in integrated_scores
    ):
        integrated_scores["neutral"] = _clip(neutral_score)
        candidate_labels.append("neutral")

    return FusionResult(
        emotion_only=rank_scores(emotion_scores, top_k=top_k),
        appraisal_only=rank_scores(appraisal_scores, top_k=top_k),
        integrated=rank_scores(integrated_scores, top_k=top_k),
        integrated_scores=integrated_scores,
        candidate_labels=candidate_labels,
        added_by_appraisal=added_by_appraisal,
    )


def compare_branches(
    emotion_scores: dict[str, float],
    appraisal_scores: dict[str, float],
    top_k: int = 5,
    emotion_weight: float = 0.80,
    appraisal_weight: float = 0.20,
    candidate_pool_size: int = 8,
    min_emotion_candidate_score: float = 0.05,
    appraisal_add_threshold: float = 0.72,
    neutral_protection_threshold: float = 0.55,
) -> dict:
    result = candidate_gated_fusion(
        emotion_scores=emotion_scores,
        appraisal_scores=appraisal_scores,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        emotion_weight=emotion_weight,
        appraisal_weight=appraisal_weight,
        min_emotion_candidate_score=min_emotion_candidate_score,
        allow_appraisal_addition=True,
        appraisal_add_threshold=appraisal_add_threshold,
        max_added_candidates=1,
        neutral_protection_threshold=neutral_protection_threshold,
    )

    return {
        "emotion_only": result.emotion_only,
        "appraisal_only": result.appraisal_only,
        "integrated": result.integrated,
        "integrated_scores": result.integrated_scores,
        "candidate_labels": result.candidate_labels,
        "added_by_appraisal": result.added_by_appraisal,
    }
