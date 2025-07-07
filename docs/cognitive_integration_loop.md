# Cognitive Integration Loop

This design note introduces a lightweight orchestration layer that binds
small language models with the existing Constraint Lattice tooling. The
loop provides a symbolic registry of available models, a simple policy
engine for routing requests, and a placeholder synthesiser for deriving
constraints from raw documents.

## Components

- **ModelRegistry** – tracks `ModelWrapper` instances keyed by name.
- **CallPolicyEngine** – chooses which model to invoke based on compute
  hints or feedback severity.
- **ConstraintSynthesizer** – converts incoming text into preliminary
  constraint structures. This stub demonstrates how unstructured
  artifacts could be transformed before being passed to
  `apply_containment_constraints` and the
  `recursive_autolearning_orchestrator` described in
  [containment_autolearning.md](containment_autolearning.md).
- **CognitiveIntegrationLoop** – coordinates model selection, processes
  input prompts, logs results to `HierarchicalMemory`, and exposes an
  asynchronous `heartbeat` for runtime status checks.

## Usage

```python
from cognitive_arch import CognitiveIntegrationLoop

loop = CognitiveIntegrationLoop()
loop.register_model("phi-2", lambda x: "phi2:" + x)
loop.register_model("gemma", lambda x: "gemma:" + x)

result = loop.process("hello", compute="low")
print(result)  # -> "phi2:hello"
```

This module is intentionally minimal. It demonstrates how Varkiel could
autonomously route requests and maintain a provenance-aware trace of its
reasoning steps.

---
Last updated: 2025-07-03
