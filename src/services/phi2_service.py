# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""PHI-2 LoRA inference wrapper.

This module offers a **lightweight** interface to run a single-call safety
classifier built on top of PHI-2 with a LoRA head.  At runtime we attempt to
load the GGUF checkpoint through ``llama_cpp``; if that's unavailable (e.g. in
CI) we fall back to a deterministic heuristic so the rest of the stack keeps
working.

Environment variables
---------------------
PHI2_MODEL_PATH   Absolute path to the *.gguf* checkpoint (default
                  *models/phi2.gguf*).
PHI2_LOG_LEVEL    Override logging level for this module.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Dict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("PHI2_LOG_LEVEL", "INFO"))

try:
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover – dev/CI will not have GPU deps
    Llama = None  # type: ignore


_DEFAULT_MODEL_PATH = os.getenv("PHI2_MODEL_PATH", "models/phi2.gguf")


@lru_cache(maxsize=1)
def _get_model():  # pragma: no cover – heavyweight
    if Llama is None:
        raise RuntimeError("llama_cpp unavailable – cannot load PHI-2 model")

    if not os.path.exists(_DEFAULT_MODEL_PATH):
        raise FileNotFoundError(_DEFAULT_MODEL_PATH)

    logger.info("Loading PHI-2 GGUF model from %s", _DEFAULT_MODEL_PATH)
    return Llama(model_path=_DEFAULT_MODEL_PATH, n_ctx=2048, logits_all=False)


def score_text(prompt: str, response: str) -> Dict[str, float]:  # noqa: D401
    """Return *confidence*/*severity* for (prompt, response).

    The fine-tuned classifier is expected to output a *risk score* token which
    we map → [0,1] confidence.  For demo we just return a stub if the heavy
    backend isn't available.
    """
    try:
        model = _get_model()
        # Very small prompt template – model returns a single float token.
        full_prompt = (
            "<CLS>\nPrompt:\n" + prompt + "\nResponse:\n" + response + "\nScore:"  # noqa: WPS336
        )
        # We ask for 1 token, which encodes a float in textual form (e.g. 0.87).
        out = model(full_prompt, max_tokens=1, stop=["\n"])
        token = out["choices"][0]["text"].strip()
        conf = float(token)
        sev = conf * 0.8
        rationale = "PHI-2 LoRA classifier"
    except Exception as exc:  # pragma: no cover – fallback path
        logger.debug("PHI-2 model unavailable: %s – using heuristic", exc)
        conf = min(len(response) / 1000.0, 1.0)
        sev = conf * 0.8
        rationale = "Length heuristic fallback"

    return {
        "confidence": conf,
        "severity": sev,
        "rationale": rationale,
    }
