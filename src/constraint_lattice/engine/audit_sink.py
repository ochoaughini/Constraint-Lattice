# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.

"""Optional Postgres/Kafka audit sinks."""

from __future__ import annotations

# If ``psycopg2`` is not installed or ``CLATTICE_PG_DSN`` is unset,
# ``save_trace`` becomes a no-op.

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS audit_traces (
    id           SERIAL PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT now(),
    trace        JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_traces_tenant_created_idx
    ON audit_traces (tenant_id, created_at DESC);
"""

import json
import os
from typing import Sequence

try:
    import psycopg2  # type: ignore
    from psycopg2.extras import execute_values  # type: ignore
except ImportError:  # pragma: no cover – optional dep
    psycopg2 = None

from .schema import AuditStep
from pipelines.trace_pipeline import publish_trace

from constraint_lattice.logging_config import configure_logger

logger = configure_logger(__name__)


def get_kafka_sink():
    """Return Kafka sink if available, else ``None``."""
    try:
        from .audit_sink_kafka import get_kafka_sink as _get
        return _get()
    except Exception:
        return None


def _get_conn():  # pragma: no cover (needs PG)
    dsn = os.getenv("CLATTICE_PG_DSN")
    if not (dsn and psycopg2):
        return None
    return psycopg2.connect(dsn)


def save_trace(trace: Sequence[AuditStep], *, tenant_id: str) -> bool:
    """Persist *trace* to Postgres.

    Returns True on success, False otherwise.
    """
    conn = _get_conn()
    trace_json = json.dumps([s.to_dict() for s in trace])

    # Always publish to Kafka (fire-and-forget)
    publish_trace({
        "tenant_id": tenant_id,
        "first_ts": trace[0].timestamp.isoformat() if trace else "",
        "last_ts": trace[-1].timestamp.isoformat() if trace else "",
        "trace_json": trace_json,
    })

    if conn is None:
        logger.debug("PG sink disabled – missing DSN or psycopg2")
        return False
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audit_traces (tenant_id, trace) VALUES (%s, %s)",
                (tenant_id, trace_json),
            )
        return True
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to write audit trace: %s", exc)
        return False
