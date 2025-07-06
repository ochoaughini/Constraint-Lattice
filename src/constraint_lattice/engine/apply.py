# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.

"""Core logic for applying constraints to LLM outputs.

Supports both sequential and batch processing with dual-mode execution
(supervisory and executor modes) and comprehensive audit tracing.

Author: Constraint Lattice Team
Last Updated: 2025-07-04
"""

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Telemetry/tracing imports
try:
    from opentelemetry import trace  # type: ignore
    _TRACER = trace.get_tracer(__name__)
except ImportError:  # pragma: no cover
    _TRACER = None  # type: ignore

from constraint_lattice.engine.telemetry import REQUEST_LATENCY_MS
from constraint_lattice.engine.score_schema import ScoreSchema
from constraint_lattice.engine.methods import METHODS
from constraint_lattice.engine.scheduler import schedule_constraints
from .agents import Fi2Agent, GemmaAgent
from .mode import get_execution_mode

# Configure logging
logger = logging.getLogger(__name__)

# Remove old logging setup
# logging.basicConfig(level=logging.INFO) - DELETE THIS LINE

@dataclass
class AuditStep:
    """Represents a single step in the constraint application process.

    Attributes:
        constraint: Name of the constraint applied
        method: Method used for application
        pre_text: Text before constraint application
        post_text: Text after constraint application
        elapsed_ms: Time taken for application (milliseconds)
        config_hash: Optional configuration hash
        tenant_id: Optional tenant ID
        model_scores: Dictionary of scores generated during application
        embeddings: Dictionary of embeddings generated during application
        timestamp: Timestamp of when the step was executed
    """
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

    # Lifecycle and core methods
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
    # Core functionality
    def to_jsonl(self, path: str) -> None:
        """Serialize audit trace to JSONL format.

        Args:
            path: Output file path
        """
        import json

        with open(path, "w", encoding="utf-8") as fh:
            for step in self:
                fh.write(json.dumps(step.to_dict()) + "\n")


# Public constraint application functions

def apply_constraints(
    prompt: str,
    output: str,
    constraints: List[Any],
    return_audit_trace: bool = False,
    **kwargs,
) -> Union[str, Tuple[str, AuditTrace]]:
    """Apply constraints to output text with dual-mode support

    This is the scalar variant for single text processing. For batched
    evaluation, use apply_constraints_batch.

    Args:
        prompt: Original prompt text
        output: Model output text to process
        constraints: List of constraints to apply
        return_audit_trace: Whether to return audit trace

    Returns:
        Processed text or tuple (text, audit_trace) if return_audit_trace is True
    """
    # Support legacy parameter name
    if "return_trace" in kwargs:
        return_audit_trace = kwargs.pop("return_trace")

    # Determine execution mode
    mode = get_execution_mode()
    
    if mode == "supervisory":
        # Supervisory mode - apply constraints as filters on LLM output
        logger.info(f"apply_constraints called with {len(constraints)} constraints")
        for i, constraint in enumerate(constraints):
            logger.info(f"Engine received constraint {i}: {constraint}, type: {type(constraint)}")

        processed_output = output
        audit_trace = AuditTrace()
        for constraint in schedule_constraints(constraints, trace=audit_trace):
            constraint_name = constraint.__class__.__name__
            method_found = False  # Track if we found a method
            try:
                for method_name, needs_prompt in METHODS.items():
                    method = getattr(constraint, method_name, None)
                    if callable(method):
                        method_found = True
                        pre_text = processed_output
                        start = datetime.now(timezone.utc)
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
                            elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000
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
                                timestamp=datetime.now(timezone.utc)
                            )
                        )
                        logger.info(
                            f"Applied {method_name} from {constraint.__class__.__name__}"
                        )
                        break
                if not method_found:
                    logger.error(
                        f"No valid method found for constraint: {constraint_name}. "
                        f"Available methods: {', '.join(METHODS.keys())}"
                    )
                    raise RuntimeError(f"No applicable method for {constraint_name}")
            except Exception as e:
                logger.error(
                    f"Error in {constraint.__class__.__name__} with prompt '{prompt}': {e}"
                )
                processed_output = output
                logger.warning("Falling back to original output due to error.")
        if return_audit_trace:
            return processed_output, audit_trace
        return processed_output
    else:
        # Executor mode - directly apply constraints using local agents
        return run_constraints_via_fi2_or_gemma(
            output=output,
            constraints=constraints,
            return_audit_trace=return_audit_trace,
            **kwargs
        )


def run_constraints_via_fi2_or_gemma(
    output: str,
    constraints: List[Any],
    return_audit_trace: bool = False,
    **kwargs,
) -> Union[str, Tuple[str, AuditTrace]]:
    """
    Execute constraints locally using fi2/gemma agents when LLM is unavailable
    """
    # Create agents
    fi2_agent = Fi2Agent()
    gemma_agent = GemmaAgent()
    
    # For now, we'll use fi2 as the default executor
    # In future we might select based on constraint types
    processed_output, steps = fi2_agent.evaluate(output, constraints)
    
    # Create audit trace
    trace = AuditTrace()
    for step in steps:
        trace.append(
            AuditStep(
                constraint=step["constraint"],
                method=step["method"],
                pre_text=step["pre_text"],
                post_text=step["post_text"],
                elapsed_ms=step["elapsed_ms"],
                timestamp=datetime.now(timezone.utc)
            )
        )
    
    if return_audit_trace:
        return processed_output, trace
    return processed_output


# Internal helper functions

def _get_batch_size(explicit: Optional[int] = None) -> Optional[int]:
    """Resolve effective batch size from parameter or environment variable.

    Args:
        explicit: Explicitly provided batch size

    Returns:
        Effective batch size to use
    """
    if explicit and explicit > 0:
        return explicit
    env_val = os.getenv("CONSTRAINT_LATTICE_BATCH_SIZE")
    if env_val and env_val.isdigit():
        return int(env_val)
    return None


# Utility functions

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


# Batched evaluation utilities

def apply_constraints_batch(
    prompts: Sequence[str],
    outputs: Sequence[str],
    constraints: Sequence[Any],
    *,
    batch_size: Optional[int] = None,
    return_trace: bool = False,
) -> Any:
    """Vectorized batch processing variant of apply_constraints.

    When JAX is enabled and constraints are JAXConstraints, uses jax.vmap
    for efficient processing. Otherwise falls back to sequential processing.

    Args:
        prompts: List of prompt texts
        outputs: List of output texts to process
        constraints: List of constraints to apply
        batch_size: Custom batch size (default: from env var)
        return_trace: Whether to return audit traces

    Returns:
        Processed texts or tuples (texts, audit_traces) if return_trace is True
    """
    from engine.jax_backend import JAXConstraint  # Local import to keep deps optional

    effective_bs = _get_batch_size(batch_size)

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
