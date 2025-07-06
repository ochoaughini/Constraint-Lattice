# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""High-volume AuditTrace pipeline: Kafka → ClickHouse.

Producer side (in-process):
  * If the env var `CLATTICE_KAFKA_BOOTSTRAP` is set, every call to
    :func:`engine.audit_sink.save_trace` additionally publishes the trace to a
    Kafka topic (default ``audit_traces``).

Consumer side (CLI / service):
  * Reads messages from Kafka, de-duplicates by the tuple
    (tenant_id, first_step_ts, last_step_ts, hash(trace_json)), and inserts into
    a ClickHouse MergeTree table partitioned by `date`
    (YYYY-MM-DD of ``first_step_ts``).

Both producer and consumer fail-open: if Kafka or ClickHouse are unreachable
metrics/logs warn but the core moderation path still succeeds.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

try:
    from kafka import KafkaProducer, KafkaConsumer  # type: ignore
except Exception:  # pragma: no cover – optional dep
    KafkaProducer = None  # type: ignore
    KafkaConsumer = None  # type: ignore

try:
    import clickhouse_connect  # type: ignore
except Exception:  # pragma: no cover
    clickhouse_connect = None  # type: ignore

_TOPIC = os.getenv("CLATTICE_KAFKA_TOPIC", "audit_traces")
_BOOTSTRAP = os.getenv("CLATTICE_KAFKA_BOOTSTRAP")
_CH_DSN = os.getenv("CLATTICE_CLICKHOUSE_DSN", "http://localhost:8123")


# ---------------------------------------------------------------------------
# Producer helper – cheap fire-and-forget
# ---------------------------------------------------------------------------

def publish_trace(message: dict) -> None:  # noqa: D401
    if not (_BOOTSTRAP and KafkaProducer):
        return
    try:
        producer = KafkaProducer(
            bootstrap_servers=_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            linger_ms=10,
        )
        producer.send(_TOPIC, message)
        producer.flush(5)
    except Exception as exc:  # pragma: no cover
        logger.warning("Kafka publish failed: %s", exc)


# ---------------------------------------------------------------------------
# Consumer service – run as separate process
# ---------------------------------------------------------------------------

@dataclass
class TraceEnvelope:
    tenant_id: str
    first_ts: str  # ISO
    last_ts: str
    trace_json: str

    @property
    def dedupe_hash(self) -> str:  # noqa: D401
        key = f"{self.tenant_id}:{self.first_ts}:{self.last_ts}:{hashlib.sha1(self.trace_json.encode()).hexdigest()}"
        return hashlib.md5(key.encode()).hexdigest()  # shorter


def run_consumer(group_id: str = "trace-sink") -> None:  # pragma: no cover
    if not (_BOOTSTRAP and KafkaConsumer and clickhouse_connect):
        raise RuntimeError("Kafka/ClickHouse deps missing or env vars not set")

    consumer = KafkaConsumer(
        _TOPIC,
        bootstrap_servers=_BOOTSTRAP,
        group_id=group_id,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )

    client = clickhouse_connect.get_client(url=_CH_DSN)

    # Create table if missing
    client.command(
        """
        CREATE TABLE IF NOT EXISTS audit_traces_ch (
            tenant_id String,
            first_ts DateTime,
            last_ts  DateTime,
            trace_json String,
            dedupe_hash FixedString(32)
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toDate(first_ts)
        ORDER BY (tenant_id, first_ts, dedupe_hash)
        """
    )

    for msg in consumer:
        data = msg.value
        env = TraceEnvelope(**data)
        # Check dedupe – rely on ReplacingMergeTree PK (dedupe_hash)
        client.insert(
            "audit_traces_ch",
            [[env.tenant_id, env.first_ts, env.last_ts, env.trace_json, env.dedupe_hash]],
            column_names=["tenant_id", "first_ts", "last_ts", "trace_json", "dedupe_hash"],
        )
        consumer.commit()
        logger.debug("Inserted trace for tenant %s", env.tenant_id)
