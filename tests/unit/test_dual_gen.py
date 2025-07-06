# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Unit test for scripts.dual_gen_demo (offline, fast)."""

import importlib
import json
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def _fake_pipe(prompt, **_):
    _fake_pipe.calls.append(prompt)
    return [{"generated_text": f"{prompt} ::fake"}]


def test_dual_generation(monkeypatch, tmp_path):
    _fake_pipe.calls = []
    # Patch the pipeline loader so no real model is loaded.
    with patch("scripts.dual_gen_demo._load_pipeline", return_value=_fake_pipe):
        demo = importlib.import_module("scripts.dual_gen_demo")
        demo._generate("hello", device="cpu")

    # Each branch should be invoked exactly twice (Gemma + Phi-2)
    assert len(_fake_pipe.calls) == 2

    traces = sorted((ROOT / "results").glob("*.trace.json"))
    assert traces, "trace file missing"
    data = json.loads(traces[-1].read_text())
    assert data, "trace empty"
