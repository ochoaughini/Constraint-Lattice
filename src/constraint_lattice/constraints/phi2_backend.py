"""Back-end providers for Phi-2 inference.

This abstraction lets us swap Hugging Face Transformers (default) for other
implementations such as **vLLM**.  It keeps `ConstraintPhi2Moderation` focused on
moderation logic while delegating language-model inference and regeneration to
back-ends that share the same tiny surface-area:

    analyse(text: str) -> dict
    regenerate(text: str, categories: list[str]) -> str

Only these two methods are required.
"""

from __future__ import annotations

from typing import Any


class Phi2Backend:  # pragma: no cover – abstract
    """Common interface for Phi-2 generation providers."""

    def analyse(self, text: str) -> dict[str, Any]:  # noqa: D401
        """Return JSON-like analysis describing policy violations."""
        raise NotImplementedError

    def regenerate(self, text: str, categories: list[str]) -> str:  # noqa: D401
        """Return a safe rewrite of *text* covering *categories* violations."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Hugging Face Transformers backend (current default)
# ---------------------------------------------------------------------------

import torch  # type: ignore
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore


class HFBackend(Phi2Backend):
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: Optional[str] = None,
        compile_model: bool = False,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)  # type: ignore[arg-type]
            except Exception:
                pass
        self.model.eval()

    # --- helpers -------------------------------------------------------- #

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:  # noqa: D401
        import json  # local import to keep top-level minimal

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return {}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}

    # --- public API ----------------------------------------------------- #

    def analyse(self, text: str) -> dict[str, Any]:  # noqa: D401
        prompt = (
            "Analyse the following text for policy violations.\n"
            "Return JSON with keys: is_safe (bool), violations (object mapping category→score),"
            "reasoning (str).\n\nText: " + text + "\n\nJSON:"
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
        out = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return self._extract_json(out)

    def regenerate(self, text: str, categories: list[str]) -> str:  # noqa: D401
        prompt = (
            "The following text was flagged for: "
            + ", ".join(categories)
            + ". Rewrite it to be safe while preserving meaning.\n\nOriginal:"
            + text
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


# ---------------------------------------------------------------------------
# JAX / Flax backend (optional)
# ---------------------------------------------------------------------------

try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
    from transformers import AutoTokenizer, FlaxAutoModelForCausalLM  # type: ignore

    class JAXBackend(Phi2Backend):
        """Phi-2 backend using Hugging Face Flax model run on JAX/XLA."""

        def __init__(
            self,
            model_name: str = "microsoft/phi-2",
            dtype: str = "float16",
            compile_model: bool = False,
            **kwargs: Any,
        ) -> None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            _dtype = {
                "float16": jnp.float16,
                "bfloat16": jnp.bfloat16,
                "float32": jnp.float32,
            }.get(dtype, jnp.float16)
            self.model = FlaxAutoModelForCausalLM.from_pretrained(
                model_name, dtype=_dtype, trust_remote_code=True
            )
            # Flax generate() works with pytree inputs.
            self._generate = self.model.generate
            if compile_model:
                self._generate = jax.jit(self._generate)  # type: ignore

        # ------------------------------ helpers ------------------------- #
        @staticmethod
        def _extract_json(text: str) -> dict[str, Any]:
            return HFBackend._extract_json(text)

        # ------------------------------ API ----------------------------- #
        def analyse(self, text: str) -> dict[str, Any]:  # noqa: D401
            prompt = (
                "Analyse the following text for policy violations.\n"
                "Return JSON with keys: is_safe (bool), violations (object mapping category→score),"
                "reasoning (str).\n\nText: " + text + "\n\nJSON:"
            )
            inputs = self.tokenizer(prompt, return_tensors="jax")
            output_ids = self._generate(
                **inputs,
                max_length=min(len(inputs["input_ids"][0]) + 128, 512),
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            ).sequences
            out = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return self._extract_json(out)

        def regenerate(self, text: str, categories: list[str]) -> str:  # noqa: D401
            prompt = (
                "The following text was flagged for: "
                + ", ".join(categories)
                + ". Rewrite it to be safe while preserving meaning.\n\nOriginal:"
                + text
                + "\n\nSafe version:"
            )
            inputs = self.tokenizer(prompt, return_tensors="jax")
            output_ids = self._generate(
                **inputs,
                max_length=min(len(inputs["input_ids"][0]) + 128, 512),
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            ).sequences
            return (
                self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                .replace(prompt, "")
                .strip()
            )

except ImportError:  # pragma: no cover

    class JAXBackend(Phi2Backend):  # type: ignore[override]
        """Stub shown when *jax*/*jaxlib* are missing."""

        def __init__(self, *args: Any, **kwargs: Any):  # noqa: D401
            raise RuntimeError(
                "JAX backend requested but the 'jax' and 'jaxlib' packages "
                "are not installed. Install with: pip install "
                "'constraint-lattice[jax]'"
            )

        def analyse(self, text: str):  # type: ignore[override]
            raise NotImplementedError

        def regenerate(self, text: str, categories):  # type: ignore[override]
            raise NotImplementedError


# ---------------------------------------------------------------------------
# vLLM backend (optional)
# ---------------------------------------------------------------------------

try:
    import threading
    import time
    from concurrent.futures import Future

    from vllm import LLM, SamplingParams  # type: ignore

    class _Batcher:
        """Background micro-batching helper for vLLM."""

        def __init__(
            self,
            llm: LLM,
            sampling_params: SamplingParams,
            max_batch: int = 8,
            timeout_ms: int = 20,
        ) -> None:
            self._llm = llm
            self._sp = sampling_params
            self._max_batch = max_batch
            self._timeout = timeout_ms / 1000.0
            self._lock = threading.Lock()
            self._cond = threading.Condition(self._lock)
            self._queue: list[tuple[str, Future]] = []
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

        def submit(self, prompt: str) -> str:
            fut: Future[str] = Future()
            with self._lock:
                self._queue.append((prompt, fut))
                if len(self._queue) >= self._max_batch:
                    self._cond.notify()
            # Wait synchronously; Constraint engine is sync today.
            return fut.result()

        def _worker(self) -> None:  # noqa: D401
            while True:
                with self._lock:
                    if not self._queue:
                        self._cond.wait(timeout=self._timeout)
                    batch = self._queue[: self._max_batch]
                    self._queue = self._queue[self._max_batch :]
                if not batch:
                    continue
                prompts = [p for p, _ in batch]
                outputs = self._llm.generate(prompts, self._sp)
                for (prompt, fut), out in zip(batch, outputs):
                    # .text contains stop token trimmed already
                    fut.set_result(out.outputs[0].text.strip())

    class VLLMBackend(Phi2Backend):
        def __init__(
            self,
            model_name: str = "microsoft/phi-2",
            gpu_memory_utilization: float = 0.90,
            max_model_len: int = 4096,
            **kwargs: Any,
        ) -> None:
            # two separate sampling param sets – analyse & regenerate
            analyse_sp = SamplingParams(
                temperature=0.3, max_tokens=128, stop=["\n"], n=1
            )
            regen_sp = SamplingParams(temperature=0.7, max_tokens=128, stop=["\n"], n=1)
            self.llm = LLM(
                model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_batched_tokens=max_model_len,
            )
            self._analyse_batcher = self._create_batcher(analyse_sp)
            self._regen_batcher = self._create_batcher(regen_sp)

        def _create_batcher(self, sp: SamplingParams) -> _Batcher:
            return self._Batcher(self.llm, sp)  # type: ignore[arg-type]

        def analyse(self, text: str) -> dict[str, Any]:  # noqa: D401
            prompt = (
                "Analyse the following text for policy violations.\n"
                "Return JSON with keys: is_safe (bool), violations (object mapping category→score),"
                "reasoning (str).\n\nText: " + text + "\n\nJSON:"
            )
            out_text = self._analyse_batcher.submit(prompt)
            return HFBackend._extract_json(out_text)

        def regenerate(self, text: str, categories: list[str]) -> str:  # noqa: D401
            prompt = (
                "The following text was flagged for: "
                + ", ".join(categories)
                + ". Rewrite it to be safe while preserving meaning.\n\nOriginal:"
                + text
                + "\n\nSafe version:"
            )
            return self._regen_batcher.submit(prompt)

except ImportError:  # pragma: no cover

    class VLLMBackend(Phi2Backend):  # type: ignore[override]
        """Stub when *vllm* is not installed."""

        def __init__(self, *args: Any, **kwargs: Any):  # noqa: D401
            raise RuntimeError(
                "vLLM backend requested but the 'vllm' package is not installed. "
                "Install with: pip install 'constraint-lattice[perf]'"
            )

        def analyse(self, text: str):  # type: ignore[override]
            raise NotImplementedError

        def regenerate(self, text: str, categories):  # type: ignore[override]
            raise NotImplementedError
