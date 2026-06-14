from __future__ import annotations

from src.appraisal_emotion_mapping import rank_scores


def integrate_emotion_scores(
    emotion_scores: dict[str, float],
    appraisal_scores: dict[str, float],
    emotion_weight: float = 0.65,
    appraisal_weight: float = 0.35,
) -> dict[str, float]:
    """
    Weighted late fusion.

    GoEmotions remains the main signal. Appraisal compatibility re-ranks or
    supports candidates without replacing the learned emotion classifier.
    """
    if emotion_weight < 0 or appraisal_weight < 0:
        raise ValueError("Fusion weights must be non-negative.")

    total_weight = emotion_weight + appraisal_weight
    if total_weight == 0:
        raise ValueError("At least one fusion weight must be positive.")

    emotion_weight /= total_weight
    appraisal_weight /= total_weight

    labels = set(emotion_scores) | set(appraisal_scores)

    return {
        label: (
            emotion_weight * float(emotion_scores.get(label, 0.0))
            + appraisal_weight * float(appraisal_scores.get(label, 0.0))
        )
        for label in labels
    }


def compare_branches(
    emotion_scores: dict[str, float],
    appraisal_scores: dict[str, float],
    top_k: int = 5,
    emotion_weight: float = 0.65,
    appraisal_weight: float = 0.35,
) -> dict:
    integrated_scores = integrate_emotion_scores(
        emotion_scores=emotion_scores,
        appraisal_scores=appraisal_scores,
        emotion_weight=emotion_weight,
        appraisal_weight=appraisal_weight,
    )

    return {
        "emotion_only": rank_scores(emotion_scores, top_k=top_k),
        "appraisal_only": rank_scores(appraisal_scores, top_k=top_k),
        "integrated": rank_scores(integrated_scores, top_k=top_k),
        "integrated_scores": integrated_scores,
    }
