from __future__ import annotations

from typing import Callable


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(*values: float) -> float:
    return sum(values) / max(len(values), 1)


def _weighted(*pairs: tuple[float, float]) -> float:
    numerator = sum(value * weight for value, weight in pairs)
    denominator = sum(weight for _, weight in pairs)
    return _clip(numerator / denominator)


def appraisal_to_emotion_scores(
    appraisal: dict[str, float],
) -> dict[str, float]:
    """
    Transparent appraisal-compatible emotion candidate rules.

    These are theory-informed heuristics, not learned emotion predictions.
    Scores represent compatibility between the appraisal profile and each
    emotion candidate.
    """
    a = {key: _clip(value) for key, value in appraisal.items()}

    pleasant = a.get("pleasantness", 0.0)
    unpleasant = a.get("unpleasantness", 0.0)
    goal_relevance = a.get("goal_relevance", 0.0)
    goal_support = a.get("goal_support", 0.0)
    suddenness = a.get("suddenness", 0.0)
    familiarity = a.get("familiarity", 0.0)
    expected_event = a.get("predict_event", 0.0)
    expected_consequence = a.get("predict_conseq", 0.0)
    urgency = a.get("urgency", 0.0)
    attention = a.get("attention", 0.0)
    effort = a.get("effort", 0.0)
    acceptance = a.get("accept_conseq", 0.0)
    self_control = a.get("self_control", 0.0)
    other_control = a.get("other_control", 0.0)
    chance_control = a.get("chance_control", 0.0)
    self_resp = a.get("self_responsblt", 0.0)
    other_resp = a.get("other_responsblt", 0.0)
    chance_resp = a.get("chance_responsblt", 0.0)
    standards = a.get("standards", 0.0)
    social_norms = a.get("social_norms", 0.0)
    not_consider = a.get("not_consider", 0.0)

    low_control = 1.0 - max(self_control, other_control, chance_control)
    uncertainty = 1.0 - _mean(expected_event, expected_consequence)
    novelty = _mean(suddenness, 1.0 - familiarity)
    obstruction = 1.0 - goal_support
    normative = _mean(standards, social_norms)
    engagement = _mean(goal_relevance, attention, 1.0 - not_consider)

    scores = {
        "anger": _weighted(
            (unpleasant, 2.0),
            (other_resp, 2.0),
            (obstruction, 1.5),
            (urgency, 1.0),
            (goal_relevance, 1.0),
        ),
        "annoyance": _weighted(
            (unpleasant, 2.0),
            (obstruction, 1.5),
            (other_resp, 1.0),
            (goal_relevance, 1.0),
        ),
        "disapproval": _weighted(
            (unpleasant, 1.0),
            (other_resp, 1.0),
            (normative, 2.0),
        ),
        "disgust": _weighted(
            (unpleasant, 2.0),
            (normative, 1.5),
            (obstruction, 0.5),
        ),
        "fear": _weighted(
            (unpleasant, 2.0),
            (uncertainty, 1.5),
            (low_control, 1.5),
            (urgency, 1.0),
            (goal_relevance, 1.0),
        ),
        "nervousness": _weighted(
            (unpleasant, 1.5),
            (uncertainty, 2.0),
            (low_control, 1.0),
            (effort, 1.0),
        ),
        "sadness": _weighted(
            (unpleasant, 2.0),
            (obstruction, 2.0),
            (low_control, 1.0),
            (acceptance, 0.5),
        ),
        "grief": _weighted(
            (unpleasant, 2.0),
            (obstruction, 2.0),
            (low_control, 1.5),
            (goal_relevance, 1.0),
            (acceptance, 0.5),
        ) * 0.85,
        "disappointment": _weighted(
            (unpleasant, 1.5),
            (obstruction, 2.0),
            (expected_event, 1.0),
            (expected_consequence, 1.0),
        ),
        "remorse": _weighted(
            (unpleasant, 1.5),
            (self_resp, 2.0),
            (normative, 1.0),
            (effort, 0.5),
        ),
        "embarrassment": _weighted(
            (unpleasant, 1.0),
            (self_resp, 1.5),
            (social_norms, 2.0),
            (attention, 0.5),
        ),
        "joy": _weighted(
            (pleasant, 2.0),
            (goal_support, 2.0),
            (goal_relevance, 1.0),
        ),
        "relief": _weighted(
            (pleasant, 1.5),
            (goal_support, 1.5),
            (acceptance, 1.0),
            (1.0 - urgency, 0.5),
        ) * 0.85,
        "gratitude": _weighted(
            (pleasant, 1.5),
            (goal_support, 1.5),
            (other_resp, 1.5),
        ) * 0.85,
        "optimism": _weighted(
            (pleasant, 1.0),
            (goal_support, 1.5),
            (expected_consequence, 1.5),
            (self_control, 1.0),
        ),
        "pride": _weighted(
            (pleasant, 1.5),
            (self_resp, 1.5),
            (goal_support, 1.0),
            (standards, 1.0),
        ),
        "surprise": _weighted(
            (suddenness, 2.0),
            (1.0 - expected_event, 1.5),
            (attention, 0.5),
        ),
        "confusion": _weighted(
            (uncertainty, 2.0),
            (1.0 - familiarity, 1.0),
            (attention, 1.0),
            (effort, 0.5),
        ),
        "curiosity": _weighted(
            (novelty, 1.5),
            (attention, 2.0),
            (goal_relevance, 0.5),
        ),
        "realization": _weighted(
            (attention, 1.0),
            (suddenness, 1.0),
            (expected_consequence, 0.5),
        ) * 0.75,
        "approval": _weighted(
            (pleasant, 1.0),
            (goal_support, 1.0),
            (normative, 1.0),
        ),
        "desire": _weighted(
            (goal_relevance, 2.0),
            (obstruction, 1.0),
            (self_control, 0.5),
            (attention, 0.5),
        ),
        "caring": _weighted(
            (engagement, 1.5),
            (pleasant, 0.5),
            (effort, 0.5),
        ) * 0.75,
        "love": _weighted(
            (pleasant, 1.5),
            (goal_support, 1.0),
            (engagement, 1.0),
        ) * 0.70,
        "excitement": _weighted(
            (pleasant, 1.5),
            (goal_relevance, 1.0),
            (urgency, 1.0),
            (suddenness, 0.5),
        ),
        "amusement": _weighted(
            (pleasant, 2.0),
            (suddenness, 0.5),
            (goal_relevance, 0.5),
        ) * 0.70,
        "admiration": _weighted(
            (pleasant, 1.0),
            (other_resp, 1.0),
            (standards, 1.0),
        ) * 0.70,
    }

    # Appraisals do not provide a reliable direct neutral rule.
    scores["neutral"] = _clip(
        1.0 - max(
            unpleasant,
            pleasant,
            goal_relevance,
            urgency,
            attention,
        )
    )

    return {
        label: _clip(score)
        for label, score in scores.items()
    }


def rank_scores(
    scores: dict[str, float],
    top_k: int = 5,
) -> list[dict[str, float]]:
    return [
        {"label": label, "score": float(score)}
        for label, score in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
    ]
