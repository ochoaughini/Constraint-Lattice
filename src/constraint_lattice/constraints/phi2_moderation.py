"""ConstraintPhi2Moderation
=================================
Advanced content-moderation constraint built on Microsoft’s Phi-2 (2.7 B) SLM.

Key features
------------
*   **Few-shot safety analysis** – categorises content into violence, hate-speech, etc.
*   **Configurable thresholds**   – per-category risk cut-offs.
*   **Fallback strategies**       – ``block`` | ``mask`` | ``regenerate``.
*   **Optional 4-bit quantisation** for ~60 % lower VRAM via bits-and-bytes.
*   **LRU result cache** to avoid repeat inference.

The class exposes a single public call‐signature that matches Constraint
Lattice’s pipeline:

>>> constraint = ConstraintPhi2Moderation()
>>> safe_text  = constraint(prompt, raw_output)
"""

from __future__ import annotations

import json
import logging

try:
    from prometheus_client import Counter, Histogram
except ImportError:  # pragma: no cover

    class _NoMetrics:
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def time(self):  # context manager
            class _NoopCtx:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    pass

            return _NoopCtx()

    Counter = Histogram = _NoMetrics  # type: ignore
import contextlib
import os
from collections import deque
from datetime import datetime
from typing import Any

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    # Lightweight stub so the module can still be imported (e.g. in CI)
    import sys
    import types

    torch = types.ModuleType("torch")  # type: ignore
    torch.cuda = types.ModuleType("cuda")  # type: ignore
    torch.cuda.is_available = lambda: False  # type: ignore
    torch.float16 = "float16"  # type: ignore
    torch.float32 = "float32"  # type: ignore
    sys.modules["torch"] = torch
    sys.modules["torch.cuda"] = torch.cuda

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except ModuleNotFoundError:  # pragma: no cover
    # Provide minimal stubs so import continues without transformers.
    import sys
    import types

    _tf_stub = types.ModuleType("transformers")
    AutoModelForCausalLM = AutoTokenizer = BitsAndBytesConfig = object  # type: ignore
    sys.modules["transformers"] = _tf_stub
    _tf_stub.AutoModelForCausalLM = AutoModelForCausalLM  # type: ignore
    _tf_stub.AutoTokenizer = AutoTokenizer  # type: ignore
    _tf_stub.BitsAndBytesConfig = BitsAndBytesConfig  # type: ignore

logger = logging.getLogger(__name__)

_REQUESTS = Counter("phi2_requests_total", "Total moderation calls")
_CACHE_HITS = Counter("phi2_cache_hits_total", "Cache hits")
_LATENCY = Histogram(
    "phi2_latency_seconds",
    "Moderation latency",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
)
logger.setLevel(os.environ.get("CLATTICE_LOG_LEVEL", "INFO"))

# --------------------------- helper utilities ---------------------------- #


from typing import Optional


def _maybe_quant_config(enable: bool) -> Optional[BitsAndBytesConfig]:
    """Return a 4-bit quantisation config if *enable* & cuda available."""
    if not enable or not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)


# ------------------------- main constraint class ------------------------ #


class ConstraintPhi2Moderation:  # pylint: disable=too-many-instance-attributes
    """Moderate content using Phi-2.

    Parameters
    ----------
    safety_thresholds : dict[str, float], optional
        Category→threshold (0-1).  Default thresholds are conservative.
    fallback_strategy : {"block", "mask", "regenerate"}, default "block"
        How to handle unsafe content.
    quantize : bool, default False
        Enable 4-bit quantisation when CUDA is present.
    cache_size : int, default 512
        Max number of recent moderation decisions to memoise.
    """

    _DEFAULT_THRESHOLDS: dict[str, float] = {
        "violence": 0.70,
        "hate_speech": 0.80,
        "harassment": 0.75,
        "self_harm": 0.90,
        "sexual_content": 0.80,
        "deception": 0.70,
    }

    _CATEGORIES: list[str] = list(_DEFAULT_THRESHOLDS)

    def __init__(
        self,
        *,
        model_name: str = "microsoft/phi-2",
        device: Optional[str] = None,
        safety_thresholds: Optional[dict[str, float]] = None,
        fallback_strategy: str = "block",
        quantize: bool = False,
        provider: str = "hf",  # 'hf' | 'vllm' | 'jax'
        compile_model: bool = False,
        cache_size: int = 512,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        provider = provider.lower()
        if provider not in {"hf", "vllm", "jax"}:
            raise ValueError("provider must be 'hf', 'vllm', or 'jax'")
        self._provider = provider

        self.compile_model = compile_model
        if fallback_strategy not in {"block", "mask", "regenerate"}:
            raise ValueError("fallback_strategy must be block | mask | regenerate")
        self.fallback_strategy = fallback_strategy

        # thresholds
        self.thresholds = {**self._DEFAULT_THRESHOLDS, **(safety_thresholds or {})}

        # backend initialisation
        logger.info("Phi-2 moderation backend=%s loading", self._provider)
        if self._provider == "vllm":
            try:
                from constraints.phi2_backend import VLLMBackend  # noqa: WPS433

                self._backend = VLLMBackend(model_name=model_name)
            except RuntimeError as exc:
                logger.warning("vLLM unavailable (%s); falling back to HFBackend", exc)
                from constraints.phi2_backend import HFBackend  # noqa: WPS433

                self._backend = HFBackend(model_name=model_name, device=self.device, compile_model=compile_model)
                self._provider = "hf"
            self.tokenizer = None  # type: ignore
            self.model = None  # type: ignore
        elif self._provider == "jax":
            from constraints.phi2_backend import JAXBackend

            self._backend = JAXBackend(
                model_name=model_name, compile_model=compile_model
            )
            self.tokenizer = None  # type: ignore
            self.model = None  # type: ignore
        else:  # hf
            from constraints.phi2_backend import HFBackend  # noqa: WPS433

            try:
                self._backend = HFBackend(model_name=model_name, device=self.device, compile_model=compile_model)
            except Exception as exc:  # pragma: no cover
                logger.warning("HFBackend unavailable (%s); using no-op stub", exc)

                class _StubBackend:  # noqa: WPS430
                    def analyse(self, text: str):
                        return {"is_safe": True, "violations": {}, "reasoning": "stub"}

                    def regenerate(self, text: str, categories: list[str]):  # noqa: D401
                        return text

                self._backend = _StubBackend()  # type: ignore
            self.tokenizer = None  # type: ignore
            self.model = None  # type: ignore
            quant_cfg = _maybe_quant_config(quantize)

        # If transformers is unavailable (CI) skip heavy initialisation.
        if self._provider in {"vllm", "jax"}:
            pass  # VLLMBackend handles its own loading
        elif hasattr(AutoTokenizer, "from_pretrained"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quant_cfg,
            )
            # optional: torch.compile for speed (PyTorch 2.2+)
            if self.compile_model and hasattr(torch, "compile"):
                try:
                    self.model = torch.compile(self.model)  # type: ignore[arg-type]
                    logger.info("Phi2Moderation: compiled model with torch.compile")
                except Exception as exc:  # pragma: no cover
                    logger.warning("torch.compile failed: %s", exc)
            self.model.eval()
            from constraints.phi2_backend import HFBackend

            self._backend = HFBackend(
                model_name=model_name, device=self.device, compile_model=compile_model
            )
        else:  # pragma: no cover
            self.tokenizer = None  # type: ignore
            self.model = None  # type: ignore
            self._backend = None

        # simple LRU cache using dict+deque
        self._cache: dict[str, str] = {}
        self._order: deque[str] = deque(maxlen=cache_size)

    # ------------------------- internal methods ------------------------ #

    def _cache_get(self, key: str) -> Optional[str]:
        val = self._cache.get(key)
        if val is not None:
            _CACHE_HITS.inc()
        if val is not None:
            with contextlib.suppress(ValueError):
                self._order.remove(key)
            self._order.append(key)
        return val

    def _cache_put(self, key: str, value: str) -> None:
        if key in self._cache:
            return
        if len(self._cache) >= self._order.maxlen:
            oldest = self._order.popleft()
            self._cache.pop(oldest, None)
        self._cache[key] = value
        self._order.append(key)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract the first JSON object embedded in *text* (best-effort)."""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return {}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}

    # -------------------------- core logic ----------------------------- #

    def _analyse(self, candidate: str) -> dict[str, Any]:
        if self._provider != "hf" and self._backend is not None:
            return self._backend.analyse(candidate)
        # default HF path (inline for speed)
        prompt = (
            "Analyse the following text for policy violations.\n"
            "Return JSON with keys: is_safe (bool), violations (object mapping category→score),"
            "reasoning (str).\n\nText: " + candidate + "\n\nJSON:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=min(len(inputs["input_ids"][0]) + 128, 512),
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return self._extract_json(text)

    def _regenerate(self, candidate: str, violations: list[str]) -> str:
        if self._provider != "hf" and self._backend is not None:
            return self._backend.regenerate(candidate, violations)
        prompt = (
            "The following text was flagged for: "
            + ", ".join(violations)
            + ". Rewrite it to be safe while preserving meaning.\n\nOriginal:"
            + candidate
            + "\n\nSafe version:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=min(len(inputs["input_ids"][0]) + 128, 512),
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return (
            self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            .replace(prompt, "")
            .strip()
        )

    # ---------------------- public pipeline API ------------------------ #

    def moderate(self, text: str) -> str:
        """Return *safe* text based on thresholds and fallback strategy."""
        _REQUESTS.inc()
        if not text.strip():
            return text
        if self._provider == "hf" and (self.model is None or self.tokenizer is None):
            # Model unavailable → act as no-op.
            return text

        key = text[:100]
        with _LATENCY.time():
            cached = self._cache_get(key)
        if cached is not None:
            return cached

        analysis = self._analyse(text)
        violations_dict = (
            analysis.get("violations", {}) if isinstance(analysis, dict) else {}
        )
        triggered = [
            c for c, s in violations_dict.items() if s >= self.thresholds.get(c, 1)
        ]
        safe = analysis.get("is_safe", False) if isinstance(analysis, dict) else False
        if safe:
            self._cache_put(key, text)
            return text

        # unsafe path
        if self.fallback_strategy == "block":
            moderated = "[Content removed due to policy violation]"
        elif self.fallback_strategy == "mask":
            moderated = "[REDACTED]"
        else:  # regenerate
            moderated = self._regenerate(text, triggered)

        self._cache_put(key, moderated)
        # optional: log audit event
        logger.info(
            "Phi2Moderation: unsafe content handled",  # will appear as message
            extra={
                "original_text": text[:120],
                "moderated_text": moderated[:120],
                "violations": triggered,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
        return moderated

    def filter_constraint(self, output: str) -> str:
        """Constraint method expected by engine.METHODS."""
        return self.moderate(output)

    def enforce_constraint(self, output: str) -> str:
        """Constraint method used by the engine (no prompt passed)."""
        return self.moderate(output)

    # allow direct call in pipelines: constraint(prompt, output)
    def __call__(self, prompt: str, output: str) -> str:  # type: ignore[override]
        # prompt is accepted for compatibility but ignored by this constraint
        return self.moderate(output)
