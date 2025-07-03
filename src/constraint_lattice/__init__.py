"""Constraint Lattice governance kernel (src layout shim).

This transitional __init__ keeps import compatibility while code is being migrated
from the legacy flat layout into the canonical ``constraint_lattice`` package
under ``src/``.  It simply re-exports the existing top-level packages (``engine``,
``constraints`` …) so both of these work:

    import engine  # legacy
    from constraint_lattice import engine  # new

Once all modules have been physically moved, these shims can be removed.
"""
from __future__ import annotations

import importlib
import sys
from types import ModuleType

__all__: list[str] = []


def _alias(top_level_name: str) -> None:
    """Import *top_level_name* and alias it under constraint_lattice.*."""
    try:
        module = importlib.import_module(top_level_name)
    except (ImportError, ModuleNotFoundError):  # optional or gated module
        return
    sys.modules[f"constraint_lattice.{top_level_name}"] = module
    setattr(sys.modules[__name__], top_level_name, module)
    __all__.append(top_level_name)


for _m in ("engine", "constraints", "sdk", "saas"):
    _alias(_m)

# Expose a PEP 561 marker so type-checkers know this is a typed pkg in future.
try:
    from importlib.metadata import version

    __version__ = version("constraint-lattice")
except Exception:  # pragma: no cover
    __version__ = "0.0.0-dev"
