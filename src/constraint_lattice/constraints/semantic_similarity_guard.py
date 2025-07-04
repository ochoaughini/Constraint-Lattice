"""SemanticSimilarityGuard constraint.

Redacts the model output when its cosine similarity to a *reference text* drops
below a threshold τ (tau).  Implements a pure-Python fallback using *numpy* and
an optional *jax* path when available.

Rationale
~~~~~~~~~
Ensures that generated text stays semantically close to an allowed reference.
This can be used, for example, to keep summaries aligned to a source document
or to prevent topic drift in multi-turn conversations.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

try:
    import jax.numpy as jnp  # type: ignore
    _HAS_JAX = True
except ModuleNotFoundError:  # pragma: no cover – CI may not have JAX
    import numpy as jnp  # type: ignore
    _HAS_JAX = False

from engine.scheduler import constraint

logger = from constraint_lattice.logging_config import configure_logger
logger = configure_logger(__name__)(__name__)

_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenise(text: str) -> List[str]:
    """Lower-case tokeniser that keeps only alphabetic words and apostrophes."""
    return _WORD_RE.findall(text.lower())


def _bow_vector(text: str):
    """Return word-count vector as a mapping token→count."""
    return Counter(_tokenise(text))


def _cosine(v1: Counter, v2: Counter) -> float:
    """Cosine similarity between two sparse vectors represented as Counters."""
    if not v1 or not v2:
        return 0.0
    keys = set(v1) | set(v2)
    dot = sum(v1[k] * v2[k] for k in keys)
    mag1 = math.sqrt(sum(n * n for n in v1.values()))
    mag2 = math.sqrt(sum(n * n for n in v2.values()))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot / (mag1 * mag2)


@constraint(priority=75, tags=["semantic", "safety"])
class SemanticSimilarityGuard:
    """Redact *output* when semantic similarity to *reference* < *tau*.

    Parameters
    ----------
    reference:
        Text against which the similarity is computed.
    tau:
        Threshold in [0, 1].  If *cosine(reference, output)* < *tau*, we blank
        the output (""), otherwise passthrough.
    """

    def __init__(self, reference: str, tau: float = 0.8):
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be between 0 and 1")
        self.reference = reference
        self.tau = tau
        self._ref_vec = _bow_vector(reference)

    # ------------------------------------------------------------------
    # Constraint entry-point expected by engine.METHODS (needs prompt)
    # ------------------------------------------------------------------
    def filter_constraint(self, prompt: str, output: str) -> str:  # noqa: D401
        """Return possibly-mutated *output*.

        If the similarity is below *tau*, the string is redacted (empty).
        We also log the score for audit / Prometheus via the global
        histogram in `engine.telemetry` when available.
        """
        sim = _cosine(self._ref_vec, _bow_vector(output))
        logger.debug("SemanticSimilarityGuard sim=%.4f tau=%.2f", sim, self.tau)

        if sim < self.tau:
            try:
                from constraint_lattice.engine.telemetry import REQUEST_ERRORS  # type: ignore
                REQUEST_ERRORS.inc()
            except Exception:  # pragma: no cover
                pass
            return ""  # Redact
        return output
