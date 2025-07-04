"""Constraint scheduling and metadata utilities.

This is a *first-pass* implementation that introduces:

1. `constraint` class decorator – attaches scheduling metadata to a
   constraint class without changing its behaviour.
2. `schedule_constraints` – orders a collection of instantiated
   constraints by priority and validates simple conflict rules.

Later milestones can replace this with a priority graph and dynamic
relevance scoring, but this gives us deterministic, explainable
precedence immediately.
"""
from __future__ import annotations

from typing import List, Sequence, Any, Callable, Dict, Protocol

class Constraint(Protocol):
    """Protocol representing a constraint instance expected by the scheduler.

    Only used for static typing / documentation – concrete constraint classes
    just need an ``enforce_constraint`` method (signature left intentionally
    loose). This avoids runtime import errors when other modules do
    ``from engine.scheduler import Constraint``.
    """

    def enforce_constraint(self, *args, **kwargs) -> Any:  # noqa: D401,E501
        ...

logger = from constraint_lattice.logging_config import configure_logger
logger = configure_logger(__name__)(__name__)

# ---------------------------------------------------------------------------
# Decorator helpers
# ---------------------------------------------------------------------------

def constraint(*, priority: int = 50, depends_on: Sequence[str] | None = None,
               conflicts_with: Sequence[str] | None = None,
               tags: Sequence[str] | None = None) -> Callable[[type], type]:
    """Attach scheduling metadata to a constraint class.

    Example:

        @constraint(priority=80, conflicts_with=["ConstraintProfanityFilter"])
        class ConstraintFoo: ...
    """
    def _decorator(cls: type) -> type:
        cls._priority = priority  # pylint: disable=protected-access
        cls._depends_on = set(depends_on or [])
        cls._conflicts_with = set(conflicts_with or [])
        cls._tags = set(tags or [])
        return cls

    return _decorator


# ---------------------------------------------------------------------------
# Scheduling algorithm (deterministic min-heap by priority)
# ---------------------------------------------------------------------------


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # noqa: D401 – type hints only
    from engine.apply import AuditTrace

def _recent_activation_penalty(constraint_name: str, trace: "AuditTrace", *, tenant_id: str | None = None) -> int:
    """Return an integer priority penalty based on recent activations in *trace*.

    The more times a constraint fired in the last 50 entries for the same
    tenant we downgrade its priority to avoid cascades.
    """
    recent = [e for e in trace[-50:] if getattr(e, "constraint", None) == constraint_name]
    if tenant_id is not None:
        recent = [e for e in recent if e.metadata.get("tenant_id") == tenant_id]
    return len(recent) * 2  # each hit lowers priority by 2

def schedule_constraints(constraints: Sequence[Any], *, trace: "AuditTrace" | None = None, tenant_id: str | None = None) -> List[Any]:
    """Return constraints ordered by *_priority* (high → low) while checking conflicts.

    If two constraints declare mutual conflict, we keep the one with the
    higher priority and drop the other, logging the decision.
    """

    name_to_obj: Dict[str, Any] = {c.__class__.__name__: c for c in constraints}

    # First, resolve conflicts – naïve pass.
    filtered: Dict[str, Any] = {}
    for name, obj in name_to_obj.items():
        conflicts = getattr(obj.__class__, "_conflicts_with", set())
        # If any conflict partner already kept with higher priority, skip.
        if any((peer in filtered) and _priority_gt(filtered[peer], obj)
               for peer in conflicts):
            logger.info("Dropping %s due to conflict precedence", name)
            continue
        filtered[name] = obj

    # Sort by priority (desc), then alphabetically for determinism.
    def _effective_key(c):
        base = getattr(c.__class__, "_priority", 50)
        penalty = 0
        if trace is not None:
            penalty = _recent_activation_penalty(c.__class__.__name__, trace, tenant_id=tenant_id)
        return (-(base - penalty), c.__class__.__name__)

    ordered = sorted(filtered.values(), key=_effective_key)
    return ordered


def _priority_gt(a: Any, b: Any) -> bool:
    return getattr(a.__class__, "_priority", 50) > getattr(b.__class__, "_priority", 50)
