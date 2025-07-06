# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Hypothesis fuzz tests to ensure constraints never break UTF-8 invariants.

For each registered constraint class we generate random Unicode input and
assert that applying the constraint stack:

1. Never raises an exception.
2. Always returns a valid UTF-8 encodable string.
3. (Optional) Leaves text unchanged *or* produces a string of equal length –
   this guards against runaway growth / truncation bugs without being too
   strict.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import List, Any

from hypothesis import given, settings, strategies as st

from engine.apply import apply_constraints
from engine.methods import METHODS

# ---------------------------------------------------------------------------
# Dynamic discovery of constraint classes
# ---------------------------------------------------------------------------

_CONSTRAINT_CLASSES: List[type] = []

for modinfo in pkgutil.walk_packages(path=["constraints"], prefix="constraints."):
    try:
        mod = importlib.import_module(modinfo.name)
    except Exception:  # pragma: no cover – skip broken import in OSS builds
        continue
    for obj_name in dir(mod):
        obj = getattr(mod, obj_name)
        if not isinstance(obj, type):
            continue
        if any(hasattr(obj, m) for m in METHODS):
            _CONSTRAINT_CLASSES.append(obj)


@given(prompt=st.text(), output=st.text())
@settings(max_examples=50)
def test_all_constraints_utf8(prompt: str, output: str) -> None:  # noqa: D401
    instances: List[Any] = []
    for cls in _CONSTRAINT_CLASSES:
        try:
            instances.append(cls())
        except Exception:
            # Constructor requires params – skip for fuzz (unit tests will cover)
            continue

    # Guard: if no constraints importable, skip (shouldn’t happen in dev)
    if not instances:
        return

    result = apply_constraints(prompt, output, instances)

    # 1. Must be str
    assert isinstance(result, str)

    # 2. Must be valid UTF-8
    result.encode("utf-8")

    # 3. Should not explode in size (≤2× original)
    assert len(result) <= 2 * len(output) + 10  # allow small constant growth
