# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Centralised telemetry initialisation.

Importing this module configures:
1. Structured JSON logging to stdout (Cloud Logging friendly).
2. OpenTelemetry tracing + metrics if `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
3. Prometheus metrics registry ready to be mounted at `/metrics` by the API layer.

Safe-to-import even if deps / env vars are missing – it will gracefully no-op.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

_JSON_FIELDS = {
    "level": "level",
    "msg": "message",
    "time": "ts",
    "logger": "logger",
}


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter suitable for Cloud Logging."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 – override
        base: dict[str, Any] = {
            _JSON_FIELDS["level"]: record.levelname,
            _JSON_FIELDS["msg"]: record.getMessage(),
            _JSON_FIELDS["time"]: datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            _JSON_FIELDS["logger"]: record.name,
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, separators=(",", ":"))


# -----------------------------------------------------------------------------
# Configure logging – called on import
# -----------------------------------------------------------------------------

_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(_JsonFormatter())

logger = from constraint_lattice.logging_config import configure_logger
logger = configure_logger(__name__)(__name__)
logger.debug("Structured logging configured.")

# -----------------------------------------------------------------------------
# OpenTelemetry (optional)
# -----------------------------------------------------------------------------

_OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
if _OTEL_ENDPOINT:
    try:
        from opentelemetry import metrics, trace  # type: ignore
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter  # type: ignore
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
        from opentelemetry.sdk.metrics import MeterProvider  # type: ignore
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore

        resource = Resource.create({"service.name": "constraint-lattice"})

        # Tracing
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=_OTEL_ENDPOINT)))
        trace.set_tracer_provider(tracer_provider)

        # Metrics
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=_OTEL_ENDPOINT),
            export_interval_millis=5000,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)

        logger.info("OpenTelemetry exporter configured at %s", _OTEL_ENDPOINT)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to init OpenTelemetry: %s", exc)

# -----------------------------------------------------------------------------
# Prometheus metrics registry
# -----------------------------------------------------------------------------

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram  # type: ignore

    REGISTRY = CollectorRegistry(auto_describe=True)

    REQUEST_LATENCY_MS = Histogram(
        "cl_request_latency_ms",
        "Request-processing latency (ms)",
        registry=REGISTRY,
    )
    REQUEST_ERRORS = Counter(
        "cl_request_errors_total",
        "Number of failed requests",
        registry=REGISTRY,
    )
except ImportError:  # pragma: no cover – prom optional
    REGISTRY = None  # type: ignore

    class _NoOp:
        def __call__(self, *args, **kwargs):
            return None

        def observe(self, *args, **kwargs):  # for Histogram
            return None

        def inc(self, *args, **kwargs):  # for Counter
            return None

    REQUEST_LATENCY_MS = _NoOp()  # type: ignore
    REQUEST_ERRORS = _NoOp()  # type: ignore


__all__ = [
    "REGISTRY",
    "REQUEST_LATENCY_MS",
    "REQUEST_ERRORS",
]
