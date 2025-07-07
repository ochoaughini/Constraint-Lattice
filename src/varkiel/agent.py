# SPDX-License-Identifier: MIT
"""Minimal orchestrator implementing Varkiel concepts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .constraint_lattice import ConstraintLattice
from .wildcore import WildCore
from .phi2_model import moderate
from .gemma_model import extract_facts
from .foundation_proxy import FoundationProxy
from .memory_store import MemoryStore
from .autolearn import DriftManager


@dataclass
class VarkielAgent:
    """Lightweight demonstration agent."""

    lattice: ConstraintLattice
    wildcore: WildCore
    foundation: FoundationProxy
    memory: MemoryStore
    drift: DriftManager

    def ingest(self, text: str, source: str = "user") -> None:
        if self.wildcore.scan(text):
            return
        text = moderate(text)
        facts = extract_facts(text)
        for i, fact in enumerate(facts):
            node_id = f"{source}_{i}"
            self.lattice.add_node(node_id, fact, source=source)
            self.memory.add(node_id, fact)
        self.drift.update(facts)

    def query(self, prompt: str) -> str:
        if self.wildcore.scan(prompt):
            return "Request blocked."
        if self.lattice.validate(prompt):
            return prompt
        external = self.foundation.query(prompt)
        if external:
            external = moderate(external)
        if external and self.lattice.validate(external):
            return external
        return "I don't know."
