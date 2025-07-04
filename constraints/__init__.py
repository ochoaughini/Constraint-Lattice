"""Compatibility shim – forwards imports to ``constraint_lattice.constraints``."""
import importlib, sys as _sys
from pathlib import Path as _Path
try:
    _target = importlib.import_module("constraint_lattice.constraints")
except ModuleNotFoundError:
    _proj_root = _Path(__file__).resolve().parent.parent
    _src_dir = _proj_root / "src"
    if _src_dir.exists():
        _sys.path.insert(0, str(_src_dir))
    _target = importlib.import_module("constraint_lattice.constraints")
_sys.modules[__name__] = _target
for _fullname, _module in list(_sys.modules.items()):
    if _fullname.startswith("constraint_lattice.constraints."):
        _alias = _fullname.replace("constraint_lattice.constraints", __name__, 1)
        _sys.modules[_alias] = _module
