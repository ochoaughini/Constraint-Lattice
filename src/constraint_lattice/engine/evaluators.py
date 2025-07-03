"""LLM-backed evaluators that return a :class:`ScoreSchema`.

These implementations are **stubs** – they demonstrate the contract and a
robust fallback chain but do NOT download large models automatically.  The
actual inference back-ends should be plugged in via environment variables or
injection at runtime.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from engine.score_schema import ScoreSchema

# Optional heavy back-ends
try:
    from services.phi2_service import score_text as _phi2_score
except Exception:  # pragma: no cover
    _phi2_score = None  # type: ignore

try:
    from services.gemma_service import classify as _gemma_classify
except Exception:  # pragma: no cover
    _gemma_classify = None  # type: ignore

logger = logging.getLogger(__name__)

__all__ = [
    "ModelEvaluator",
    "NullEvaluator",
    "Phi2Evaluator",
    "GemmaEvaluator",
    "FallbackEvaluator",
]


class ModelEvaluator:  # pragma: no cover – Abstract baseline
    name: str = "base"

    async def score(self, *, prompt: str, response: str, metadata: Dict[str, Any]) -> ScoreSchema:  # noqa: D401
        raise NotImplementedError


class NullEvaluator(ModelEvaluator):
    """Deterministic evaluator that returns zero-risk scores."""

    name = "null"

    async def score(self, *, prompt: str, response: str, metadata: Dict[str, Any]) -> ScoreSchema:  # noqa: D401
        return ScoreSchema(confidence=0.0, severity=0.0, rationale="Null evaluator – no signal")


class Phi2Evaluator(ModelEvaluator):
    """Stub that simulates PHI-2 classifier output.

    In production replace the random score with an actual inference call
    (e.g. through HuggingFace transformers or llama-cpp with a LoRA head).
    """

    name = "phi2"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    async def score(self, *, prompt: str, response: str, metadata: Dict[str, Any]) -> ScoreSchema:  # noqa: D401
        if _phi2_score is not None:
            data = _phi2_score(prompt, response)
            return ScoreSchema(**data)
        # Fallback heuristic
        conf = min(len(response) / 1000.0, 1.0)
        sev = conf * 0.8
        rationale = "Heuristic length-based PHI-2 fallback"
        return ScoreSchema(confidence=conf, severity=sev, rationale=rationale)


class GemmaEvaluator(ModelEvaluator):
    """Stub evaluator for Gemma embeddings + classifier."""

    name = "gemma"

    async def score(self, *, prompt: str, response: str, metadata: Dict[str, Any]) -> ScoreSchema:  # noqa: D401
        if _gemma_classify is not None:
            data = _gemma_classify(prompt, response)
            return ScoreSchema(**data)
        # Stub constant
        return ScoreSchema(confidence=0.3, severity=0.2, rationale="Stub Gemma score")


class FallbackEvaluator(ModelEvaluator):
    """Try multiple evaluators in order until one succeeds."""

    name = "fallback"

    def __init__(self, evaluators: Optional[list[ModelEvaluator]] = None):
        if evaluators is None:
            evaluators = [GemmaEvaluator(), Phi2Evaluator(), NullEvaluator()]
        self.evaluators = evaluators

    async def score(self, *, prompt: str, response: str, metadata: Dict[str, Any]) -> ScoreSchema:  # noqa: D401
        last_exc: Exception | None = None
        for ev in self.evaluators:
            try:
                return await ev.score(prompt=prompt, response=response, metadata=metadata)
            except Exception as exc:  # pragma: no cover
                logger.warning("%s evaluator failed: %s", ev.name, exc)
                last_exc = exc
        # If all fail, return deterministic null score.
        logger.error("All evaluators failed, falling back to NullEvaluator: %s", last_exc)
        return await NullEvaluator().score(prompt=prompt, response=response, metadata=metadata)
