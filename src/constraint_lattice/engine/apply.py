import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Dict
from contextlib import nullcontext

# telemetry / tracing
try:
    from opentelemetry import trace  # type: ignore
    _TRACER = trace.get_tracer(__name__)
except Exception:  # pragma: no cover
    _TRACER = None  # type: ignore

from constraint_lattice.engine.telemetry import REQUEST_LATENCY_MS
from constraint_lattice.engine.score_schema import ScoreSchema

from constraint_lattice.engine.methods import METHODS
from constraint_lattice.engine.scheduler import schedule_constraints

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AuditStep:
    constraint: str
    method: str
    pre_text: str
    post_text: str
    elapsed_ms: float
    config_hash: Optional[str] = None
    tenant_id: Optional[str] = None
    model_scores: Dict[str, float] = field(default_factory=dict)
    embeddings: Dict[str, list] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self):
        """Convert AuditStep to a dictionary with serializable values."""
        base = {
            "constraint": self.constraint,
            "method": self.method,
            "pre_text": self.pre_text,
            "post_text": self.post_text,
            "elapsed_ms": self.elapsed_ms,
            "timestamp": self.timestamp.isoformat(),
        }
        # Optional additions
        if self.config_hash:
            base["config_hash"] = self.config_hash
        if self.tenant_id:
            base["tenant_id"] = self.tenant_id
        if self.model_scores:
            base["model_scores"] = self.model_scores
        if self.embeddings:
            base["embeddings"] = self.embeddings
        return base


class AuditTrace(list):
    def to_jsonl(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as fh:
            for step in self:
                fh.write(json.dumps(step.to_dict()) + "\n")


def apply_constraints(
    prompt: str, output: str, constraints: list[Any], *, return_trace: bool = False
) -> Any:
    """Apply *one by one* each constraint to the ``output`` string.

    This is the **scalar** variant – kept for backward-compatibility and CLI
    simplicity.  For batched evaluation (multiple outputs/prompts) see
    :pyfunc:`apply_constraints_batch`.
    """
    processed_output = output
    audit_trace = AuditTrace()
    for constraint in schedule_constraints(constraints, trace=audit_trace):
        constraint_name = constraint.__class__.__name__
        try:
            for method_name, needs_prompt in METHODS.items():
                method = getattr(constraint, method_name, None)
                if callable(method):
                    pre_text = processed_output
                    start = datetime.utcnow()
                    span_ctx = (
                        _TRACER.start_as_current_span(f"{constraint_name}.{method_name}")  # type: ignore
                        if _TRACER is not None
                        else nullcontext()
                    )
                    with span_ctx as span:  # type: ignore
                        if needs_prompt:
                            processed_output = method(prompt, processed_output)
                        else:
                            processed_output = method(processed_output)
                        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
                        # metrics / span attrs
                        try:
                            if REQUEST_LATENCY_MS:
                                REQUEST_LATENCY_MS.observe(elapsed)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        if span:
                            span.set_attribute("constraint", constraint_name)
                            span.set_attribute("method", method_name)
                            span.set_attribute("elapsed_ms", elapsed)
                    audit_trace.append(
                        AuditStep(
                            constraint=constraint_name,
                            method=method_name,
                            pre_text=pre_text,
                            post_text=processed_output,
                            elapsed_ms=elapsed,
                        )
                    )
                    logger.info(
                        f"Applied {method_name} from {constraint.__class__.__name__}"
                    )
                    break
        except Exception as e:
            logger.error(
                f"Error in {constraint.__class__.__name__} with prompt '{prompt}': {e}"
            )
            processed_output = output
            logger.warning("Falling back to original output due to error.")
    if return_trace:
        return processed_output, audit_trace
    return processed_output


# ---------------------------------------------------------------------------
# Batched evaluation utilities (experimental)
# ---------------------------------------------------------------------------

from typing import Sequence, Tuple, Optional
import itertools
import os


def _get_batch_size(explicit: Optional[int] = None) -> Optional[int]:
    """Resolve effective batch size from parameter or env var CONSTRAINT_LATTICE_BATCH_SIZE."""
    if explicit and explicit > 0:
        return explicit
    env_val = os.getenv("CONSTRAINT_LATTICE_BATCH_SIZE")
    if env_val and env_val.isdigit():
        return int(env_val)
    return None


def apply_constraints_batch(
    prompts: Sequence[str],
    outputs: Sequence[str],
    constraints: Sequence[Any],
    *,
    batch_size: Optional[int] = None,
    return_trace: bool = False,
):
    """Vectorised variant of :func:`apply_constraints`.

    When JAX is enabled and a constraint is an instance of :class:`engine.jax_backend.JAXConstraint`,
    we evaluate it in a single `jax.vmap` pass. Otherwise we fall back to the
    scalar loop. The function optionally chunks the batch to control memory via
    `batch_size` or the `CONSTRAINT_LATTICE_BATCH_SIZE` env var.
    """
    from engine.jax_backend import JAXConstraint  # Local import to keep deps optional

    effective_bs = _get_batch_size(batch_size)

    # Helper that yields chunked views on the data
    def _chunk(seq, n):
        if not n:
            yield list(seq)
            return
        it = iter(seq)
        while True:
            chunk = list(itertools.islice(it, n))
            if not chunk:
                break
            yield chunk

    prompt_chunks = list(_chunk(prompts, effective_bs))
    output_chunks = list(_chunk(outputs, effective_bs))

    full_trace = AuditTrace()
    processed_outputs: list[str] = []

    for p_chunk, o_chunk in zip(prompt_chunks, output_chunks):
        # Ensure lists for mutability
        o_chunk = list(o_chunk)
        for constraint in schedule_constraints(constraints, trace=full_trace):
            if isinstance(constraint, JAXConstraint):
                try:
                    import jax.numpy as jnp  # pylint: disable=import-error

                    arr = jnp.asarray(o_chunk)
                    mask = constraint(arr, batch=True)
                    # Convert mask to Python bools
                    mask_py = [bool(m) for m in mask]
                    o_chunk = [out if passed else "" for out, passed in zip(o_chunk, mask_py)]
                    continue  # Move to next constraint after vectorised run
                except Exception as exc:  # pragma: no cover
                    logger.warning("JAX batch evaluation failed: %s; falling back to scalar", exc)
            # Scalar fallback per sample
            for idx, (pr, out) in enumerate(zip(p_chunk, o_chunk)):
                new_out, trace = apply_constraints(pr, out, [constraint], return_trace=True)
                o_chunk[idx] = new_out
                full_trace.extend(trace)
        processed_outputs.extend(o_chunk)

    if return_trace:
        return processed_outputs, full_trace
    return processed_outputs
