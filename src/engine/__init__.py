# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Compatibility shim to keep legacy `import engine.*` working after the
package was moved under the canonical `constraint_lattice.engine` namespace.

This module forwards all imports and attribute access to
`constraint_lattice.engine` so that both of these continue to succeed::

    import engine.apply  # legacy
    from constraint_lattice.engine import apply  # canonical
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType

# Import the canonical implementation and alias it.
_target: ModuleType = importlib.import_module("constraint_lattice.engine")

# Expose the submodule search locations so that ``import engine.<sub>`` works
# via the normal import machinery.
__path__ = _target.__path__  # type: ignore[attr-defined]  # noqa: D401 – re-export

# Register the alias in ``sys.modules`` *before* we start populating children
# to avoid infinite recursion if any of the imports below reference us.
sys.modules[__name__] = _target

# ---------------------------------------------------------------------------
# Eagerly alias already-imported submodules so that tests relying on direct
# subpackage imports (e.g. ``import engine.apply``) succeed even if those
# modules were imported *before* this shim.
# ---------------------------------------------------------------------------
for _modname, _module in list(sys.modules.items()):
    if _modname.startswith("constraint_lattice.engine."):
        _alias = _modname.replace("constraint_lattice.engine", __name__, 1)
        sys.modules.setdefault(_alias, _module)

# ---------------------------------------------------------------------------
# Lazy loader fallback – if a submodule is imported that we have not aliased
# yet, import it from the canonical namespace on demand.
# ---------------------------------------------------------------------------
import importlib.util as _iu

def __getattr__(name: str):  # noqa: D401 – PEP 562 dynamic attr loader
    full_name = f"constraint_lattice.engine.{name}"
    try:
        return importlib.import_module(full_name)
    except ModuleNotFoundError as err:
        raise AttributeError(name) from err

# Mark the module as package-typed for static analyzers.
__all__: list[str] = []
