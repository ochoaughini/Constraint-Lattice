# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.


"""Compatibility shim – forwards imports to ``constraint_lattice.engine``."""
import importlib, sys as _sys
from pathlib import Path as _Path
try:
    _target = importlib.import_module("constraint_lattice.engine")
except ModuleNotFoundError:
    # If running from source checkout without editable install, add ./src to path.
    _proj_root = _Path(__file__).resolve().parent.parent
    _src_dir = _proj_root / "src"
    if _src_dir.exists():
        _sys.path.insert(0, str(_src_dir))
    _target = importlib.import_module("constraint_lattice.engine")
# Mirror package characteristics.
# Merge canonical package search path so submodule discovery works even
# when editable installs or PYTHONPATH shims are missing.
try:
    from importlib.machinery import ModuleSpec
    __path__ = list(_target.__path__)  # type: ignore[attr-defined]
    _src_engine_dir = (_Path(__file__).parent.parent / "src" / "constraint_lattice" / "engine").resolve()
    if str(_src_engine_dir) not in __path__ and _src_engine_dir.exists():
        __path__.append(str(_src_engine_dir))
except Exception:  # pragma: no cover
    __path__ = _target.__path__  # type: ignore
_sys.modules[__name__] = _target
# Re-export submodules so ``import engine.apply`` still works.
import pkgutil as _pu
# Preload and alias every submodule so standard import semantics succeed.
for _finder, _modname, _ispkg in _pu.walk_packages(_target.__path__, prefix="constraint_lattice.engine."):
    try:
        _sub = importlib.import_module(_modname)
    except Exception:
        continue
    _alias = _modname.replace("constraint_lattice.engine", __name__, 1)
    _sys.modules[_alias] = _sub

# Fallback dynamic loader for not-yet-imported subpackages
import importlib as _il

def __getattr__(name: str):  # noqa: D401 – PEP 562 lazy attr loader
    try:
        return _il.import_module(f"constraint_lattice.engine.{name}")
    except ModuleNotFoundError as _err:
        raise AttributeError(name) from _err
