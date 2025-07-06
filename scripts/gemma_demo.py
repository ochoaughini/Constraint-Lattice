# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""gemma_demo.py

Download Google Gemma-2B-IT via Hugging Face *transformers*, generate a quick
response, run it through Constraint Lattice, and print the moderated text and
AuditTrace.  Meant for manual experimentation — **not** included in CI because
it requires ~4 GB of weights.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import pipeline  # type: ignore

from sdk.engine import ConstraintEngine

PRIMARY_MODEL = "google/gemma-2-2b-it"  # gated but preferred
FALLBACK_MODEL = "gpt2-medium"
CACHE_DIR = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

from huggingface_hub.utils import GatedRepoError


def _load_pipeline(model_id: str):
    try:
        return pipeline(
            "text-generation",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1,
        )
    except (GatedRepoError, OSError, Exception) as exc:
        raise RuntimeError(f"Model {model_id} unavailable: {exc}") from exc


def generate_raw(prompt: str) -> str:
    """Generate text, preferring Gemma but falling back to GPT-2 if gated."""
    for model_id in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            pipe = _load_pipeline(model_id)
            break
        except RuntimeError as err:
            print(err)
            continue
    else:
        raise RuntimeError("No available generation model found.")

    result: list[dict] = pipe(prompt, max_new_tokens=64, do_sample=False)
    return result[0]["generated_text"]  # type: ignore[index]


def main() -> None:
    prompt = "Who are you?"
    print("Attempting Gemma; will fall back to GPT-2 if access is gated…")
    raw = generate_raw(prompt)
    print("\n----- RAW MODEL OUTPUT -----\n", raw)

    # Run through Constraint Lattice
    engine = ConstraintEngine()
    moderated, trace = engine.run(prompt, raw, return_trace=True)
    print("\n----- MODERATED OUTPUT -----\n", moderated)

    print("\n----- AUDIT TRACE -----")
    for step in trace:
        print(step.to_dict())


if __name__ == "__main__":
    main()
