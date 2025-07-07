# Varkiel Architecture Overview

This document summarizes the Varkiel cognitive agent design as implemented in this repository. Varkiel operates in **offline** and **online** modes while enforcing a lattice of constraints to avoid hallucinations and policy violations.

## Core Components

- **Core Orchestrator** – entry point that coordinates reasoning steps and transitions between phases.
- **Constraint Lattice Engine** – stores facts and rules as nodes in a lattice. All new content is validated against this structure.
- **WildCore Detector** – anomaly detection layer used on all inputs and candidate outputs.
- **Local Inference Cores** – lightweight models (Phi‑2 and Gemma) used for moderation and formalization when no internet is available.
- **Foundation Model Interface** – optional proxy used in online mode to query external LLMs. Returned answers are validated before use.
- **Semantic Memory Store** – persistent knowledge base indexed by embeddings and symbolic keys.
- **Autonomous Learning** – background process monitoring drift and updating constraints over time.

## Offline Mode

When running without internet access Varkiel relies exclusively on local resources:

1. Input is moderated with Phi‑2 and parsed by Gemma.
2. Parsed knowledge is inserted as new nodes in the Constraint Lattice.
3. Queries are answered using only stored constraints and local inference. Any draft answer must be verified by the lattice before delivery.
4. If verification fails, Varkiel refuses or requests clarification instead of guessing.

## Online Mode

In online mode the orchestrator may delegate sub‑tasks to external LLMs:

1. The Core formulates a focused epistemic query including relevant constraints.
2. The Foundation Model Interface sends this prompt to one or more models (e.g. GPT‑4 or Claude).
3. Model outputs are evaluated by the lattice and WildCore. Conflicting or unverifiable statements are rejected.
4. Varkiel may integrate verified new knowledge into the memory store and iteratively refine the response.

The constraint lattice ensures that online assistance cannot override existing policies or introduce hallucinations.

## Files

- `varkiel_agent_main/src/varkiel/central_controller.py` – main orchestrator implementation.
- `varkiel_agent_main/src/varkiel/constraint_lattice_adapter.py` – interface to the constraint lattice.
- `varkiel_agent_main/ARCHITECTURE.md` – original high‑level blueprint.

This overview complements those documents with a concise explanation of offline and online operation.
