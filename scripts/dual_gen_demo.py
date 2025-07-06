# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
#!/usr/bin/env python
"""Gemma vs Phi-2 dual generation demo (offline-friendly).

This CLI produces three outputs for every prompt:
1. Gemma raw
2. Gemma moderated by Constraint Lattice using Phi-2
3. Phi-2 raw

The first run downloads model weights into ``hf-cache/``. Subsequent runs are
completely offline if ``HF_HUB_OFFLINE=1`` is set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import contextlib

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = AutoTokenizer = None  # type: ignore

    def pipeline(*args, **kwargs):  # type: ignore
        raise RuntimeError("transformers not installed; install with pip install transformers")


from constraints.phi2_moderation import ConstraintPhi2Moderation
from sdk.engine import ConstraintEngine

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "hf-cache"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PRIMARY_MODEL = "google/gemma-2-2b-it"
FALLBACK_MODEL = "microsoft/phi-2"

# Fixed-width table rendering helpers
TABLE_WIDTH = 76
SEP_LINE = "─" * (TABLE_WIDTH + 18)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_pipeline(model_id: str, device: str):
    """Return a transformers text-generation pipeline for *model_id*."""
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = True

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device=device,
        trust_remote_code=True,
    )


# ---------------------------------------------------------------------------
# Core generation routine
# ---------------------------------------------------------------------------


def _generate(prompt: str, device: str):
    # Try Gemma first; if gated/unavailable, fall back.
    for model_name in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            pipe = _load_pipeline(model_name, device)
            break
        except Exception as exc:
            print(f"[warn] could not load {model_name}: {exc}", file=sys.stderr)
            if model_name == FALLBACK_MODEL:
                raise
            continue

    _inf_ctx = (
        getattr(torch, "inference_mode", None)
        or getattr(torch, "no_grad", None)
        or contextlib.nullcontext
    )
    with _inf_ctx():
        gemma_raw = pipe(prompt, max_new_tokens=64, do_sample=False)[0][
            "generated_text"
        ]
        phi2_raw = pipe(prompt, max_new_tokens=64, do_sample=False)[0]["generated_text"]

    # Single moderation instance
    engine: ConstraintEngine
    try:
        engine = ConstraintEngine(extra_constraints=[ConstraintPhi2Moderation(device=device)])  # type: ignore[arg-type]
    except TypeError:
        # Older SDK version without extra_constraints support
        engine = ConstraintEngine()
    gemma_mod = engine.run(prompt, gemma_raw)

    # Persist audit trace
    trace_path = RESULTS_DIR / f"{int(time.time())}.trace.json"
    trace_data = getattr(engine, "last_trace", None)
    if not trace_data:
        trace_data = {
            "prompt": prompt,
            "gemma_raw": gemma_raw,
            "gemma_moderated": gemma_mod,
            "phi2_raw": phi2_raw,
        }
    trace_path.write_text(json.dumps(trace_data, indent=2))

    _render_table(gemma_raw, gemma_mod, phi2_raw)
    print(f"[trace] saved → {trace_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _render_table(g_raw: str, g_mod: str, p_raw: str):
    def row(label: str, text: str):
        print(f"{label:<15}│ {text[:TABLE_WIDTH]}")

    print(SEP_LINE)
    row("Gemma raw", g_raw)
    row("Gemma moderated", g_mod)
    row("Phi-2 raw", p_raw)
    print(SEP_LINE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    ap = argparse.ArgumentParser(description="Gemma vs Phi-2 generation demo")
    ap.add_argument("--prompt", "-p", help="Prompt string. If omitted, stdin is read.")
    ap.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:0")
    return ap.parse_args()


def main():
    args = _parse_args()
    prompt = args.prompt or sys.stdin.read()
    if not prompt.strip():
        sys.exit("[error] prompt cannot be empty")
    _generate(prompt, args.device)


if __name__ == "__main__":
    main()
