"""FastAPI dependency helpers for tenant resolution, RBAC, and rate-limits."""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List

from fastapi import Depends, Header, HTTPException, Request

from engine.loader import load_constraints_from_yaml
from engine.tenant_policy import TenantPolicy
from engine.evaluators import FallbackEvaluator

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Very small in-process caches – in production these would live in Redis.
# ----------------------------------------------------------------------------

_REQUEST_HISTORY: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=1000))
_POLICY_CACHE: Dict[str, TenantPolicy] = {}

_DEFAULT_RPS = int(os.getenv("CLATTICE_DEFAULT_RPS", "60"))
_CONSTRAINTS_YAML = os.getenv("CLATTICE_CONSTRAINTS_PATH", "constraints.yaml")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _bucket_ok(tenant_id: str, limit: int) -> bool:
    """Token-bucket style check – allow if requests in last 60s < limit."""
    now = time.time()
    hist = _REQUEST_HISTORY[tenant_id]
    # Evict old timestamps >60s
    while hist and now - hist[0] > 60:
        hist.popleft()
    if len(hist) >= limit:
        return False
    hist.append(now)
    return True


# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------


def get_tenant_id(x_api_key: str = Header(..., alias="X-API-Key")) -> str:  # noqa: D401
    """Derive tenant id from API key – here we simply hash the key."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    return str(abs(hash(x_api_key)) % (10**8))  # pseudo-id


def rate_limit(tenant_id: str = Depends(get_tenant_id)) -> None:  # noqa: D401
    """Enforce per-tenant requests/minute limits defined in TenantPolicy."""
    policy = _POLICY_CACHE.get(tenant_id)
    limit = policy.rps_limit if policy else _DEFAULT_RPS
    if not _bucket_ok(tenant_id, limit):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


async def get_tenant_policy(tenant_id: str = Depends(get_tenant_id)) -> TenantPolicy:  # noqa: D401
    """Fetch or build a TenantPolicy for *tenant_id*."""
    if tenant_id in _POLICY_CACHE:
        return _POLICY_CACHE[tenant_id]

    # For demo we load default constraint profile; real impl would query DB.
    constraints = load_constraints_from_yaml(_CONSTRAINTS_YAML, profile="default", search_modules=[])

    policy = TenantPolicy(
        tenant_id=tenant_id,
        constraints=constraints,
        evaluators=[FallbackEvaluator()],
        rps_limit=_DEFAULT_RPS,
        scopes=frozenset({"default"}),
    )

    _POLICY_CACHE[tenant_id] = policy
    return policy


async def enforce_scope(scope: str, policy: TenantPolicy = Depends(get_tenant_policy)) -> None:  # noqa: D401
    if scope not in policy.scopes:
        raise HTTPException(status_code=403, detail="Missing required scope")
