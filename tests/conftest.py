# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Test configuration – ensures project root is on ``sys.path``.

Pytest launches test collection from within the repository root, but IDEs or
certain CI layouts might change the working directory.  To guarantee that the
package directories (``engine``, ``constraints``, ``sdk``) are importable even
when the root isn't first on ``sys.path``, we explicitly insert it.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (str(SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Also expose bundled example packages so imports succeed without installation.
EXTRA_PKGS = [
    ROOT / "wild_core_main" / "src",
    ROOT / "varkiel_agent_main" / "src",
]
for _p in map(str, EXTRA_PKGS):
    if _p not in sys.path and pathlib.Path(_p).exists():
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Lightweight stubs for heavy optional ML back-ends so that importing modules
# that *reference* them (e.g. transformers with Flax / JAX) will not error.
# ---------------------------------------------------------------------------
import types

for _name in [
    "flax",
    "jax",
    "jaxlib",
    "torch_xla",
]:
    if _name not in sys.modules:
        mod = types.ModuleType(_name)
        import importlib.machinery as _mach

        mod.__spec__ = _mach.ModuleSpec(name=_name, loader=None)
        mod.__path__ = []  # mark as namespace package
        sys.modules[_name] = mod

# Ensure nested import paths like 'jax.numpy' resolve to numpy fallback
import numpy as _np

_jax = sys.modules.get("jax")
if _jax is not None and not hasattr(_jax, "numpy"):
    _np_sub = types.ModuleType("jax.numpy")
    _np_sub.__dict__.update(_np.__dict__)
    import importlib.machinery as _mach

    _np_sub.__spec__ = _mach.ModuleSpec(name="jax.numpy", loader=None)
    _np_sub.__path__ = []
    sys.modules["jax.numpy"] = _np_sub
    setattr(_jax, "numpy", _np_sub)

# Hint to transformers to skip missing back-ends.
import os

os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
