"""Load constraint scheduling policy from declarative YAML.

Admins can ship hot-fixes by editing `policies/constraints.yaml` without a
code deploy.  The spec supports:

- enable / disable constraints
- set priority
- express `after` dependencies

Example YAML::

    - name: ConstraintProfanityFilter
      enabled: true
      priority: 90
      after: []

If a listed constraint cannot be imported we log a warning and skip it.
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import List, Sequence, Type

import yaml  # type: ignore

from engine.scheduler import Constraint  # re-export for type hints

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "policies/constraints.yaml"


def _import_constraint(name: str) -> Type[Constraint] | None:
    try:
        module_name, class_name = name.rsplit(".", 1)
    except ValueError:  # no dot – assume constraints.<name>
        module_name = f"constraints.{name.lower()}"
        class_name = name
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)  # type: ignore[return-value]
    except Exception as exc:  # pragma: no cover
        logger.warning("Policy loader: could not import %s – %s", name, exc)
        return None


def load_constraints(policy_path: Path | None = None) -> List[Constraint]:
    path = policy_path or _DEFAULT_PATH
    if not path.exists():
        logger.info("No policy file found at %s – skipping", path)
        return []
    doc = yaml.safe_load(path.read_text()) or []
    constraints: List[Constraint] = []
    for entry in doc:
        if not entry.get("enabled", True):
            continue
        c_cls = _import_constraint(entry["name"])
        if not c_cls:
            continue
        kwargs = {}
        if "mode" in entry:
            kwargs["mode"] = entry["mode"]
        cons = c_cls(**kwargs)
        # Override scheduler metadata
        cons.priority = entry.get("priority", cons.priority)  # type: ignore[attr-defined]
        if after := entry.get("after"):
            cons.after = frozenset(after)  # type: ignore[attr-defined]
        constraints.append(cons)
    return constraints
