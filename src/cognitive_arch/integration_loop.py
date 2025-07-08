from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .hierarchical_memory import HierarchicalMemory
from .emancipation_metric import EmancipationMetric


class ModelWrapper:
    """Lightweight wrapper around a callable model."""

    def __init__(
        self, name: str, call_fn: Callable[[str], str], threshold: float = 1.0
    ) -> None:
        self.name = name
        self.call_fn = call_fn
        self.threshold = threshold

    def __call__(self, prompt: str) -> str:
        return self.call_fn(prompt)


@dataclass
class ModelRegistry:
    """Registry of available models."""

    models: dict[str, ModelWrapper] = field(default_factory=dict)

    def register(self, model: ModelWrapper) -> None:
        self.models[model.name] = model

    def get(self, name: str) -> ModelWrapper | None:
        return self.models.get(name)


class CallPolicyEngine:
    """Simple policy engine to select a model based on heuristics."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def select(self, compute: str | None = None, severity: float = 0.0) -> ModelWrapper:
        if compute == "low" and "phi-2" in self.registry.models:
            return self.registry.models["phi-2"]
        if severity > 0.5 and "gemma" in self.registry.models:
            return self.registry.models["gemma"]
        return next(iter(self.registry.models.values()))


class ConstraintSynthesizer:
    """Placeholder for converting raw text into constraint representations."""

    def synthesize(self, text: str) -> dict[str, Any]:
        return {"constraint": text[:30]}


class CognitiveIntegrationLoop:
    """Coordinator tying together models, policy, and memory."""

    def __init__(self, memory_path: str = "memory.json") -> None:
        self.registry = ModelRegistry()
        self.policy = CallPolicyEngine(self.registry)
        self.synthesizer = ConstraintSynthesizer()
        self.memory = HierarchicalMemory(memory_path)
        self.emancipation = EmancipationMetric(self.memory)

    def register_model(self, name: str, call_fn: Callable[[str], str]) -> None:
        self.registry.register(ModelWrapper(name, call_fn))

    def process(
        self, prompt: str, compute: str | None = None, severity: float = 0.0
    ) -> str:
        model = self.policy.select(compute, severity)
        result = model(prompt)
        self.memory.add(["trace", model.name], {"prompt": prompt, "result": result})
        # Update emancipation score based on output length ratio
        score = len(result) / max(len(prompt), 1)
        self.emancipation.update(min(score, 1.0))
        return result

    async def heartbeat(self, interval: float = 60.0) -> None:
        while True:
            await asyncio.sleep(interval)
            status = list(self.registry.models.keys())
            self.memory.add(["heartbeat"], {"registered_models": status})
