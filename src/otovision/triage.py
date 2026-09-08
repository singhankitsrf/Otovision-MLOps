from __future__ import annotations

import math
import numpy as np


def normalized_entropy(probabilities) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    entropy = -np.sum(p * np.log(p))
    return float(entropy / math.log(len(p)))


def triage_decision(
    probabilities, min_confidence: float = 0.75, max_normalized_entropy: float = 0.65
):
    p = np.asarray(probabilities, dtype=float)
    confidence = float(p.max())
    entropy = normalized_entropy(p)
    review_required = confidence < min_confidence or entropy > max_normalized_entropy
    return {
        "confidence": confidence,
        "normalized_entropy": entropy,
        "review_required": bool(review_required),
        "decision": "human_review" if review_required else "model_assisted_prediction",
    }
