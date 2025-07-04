"""Compatibility shim for legacy ``import constraints.*`` imports.

For backward-compatibility we expose the canonical
``constraint_lattice.constraints`` package under the top-level
``constraints`` namespace.
"""
from __future__ import annotations

import importlib
import sys
from types import ModuleType

_target: ModuleType = importlib.import_module("constraint_lattice.constraints")

__path__ = _target.__path__  # type: ignore[attr-defined]

sys.modules[__name__] = _target

# Eager alias for already-loaded submodules.
for _modname, _module in list(sys.modules.items()):
    if _modname.startswith("constraint_lattice.constraints."):
        _alias = _modname.replace("constraint_lattice.constraints", __name__, 1)
        sys.modules.setdefault(_alias, _module)

import importlib.util as _iu

def __getattr__(name: str):  # PEP 562 dynamic loader
    full_name = f"constraint_lattice.constraints.{name}"
    try:
        return importlib.import_module(full_name)
    except ModuleNotFoundError as err:
        raise AttributeError(name) from err

__all__: list[str] = []
