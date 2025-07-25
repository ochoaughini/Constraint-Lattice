"""Lightweight cognitive architecture components."""

from .hierarchical_memory import HierarchicalMemory
from .metacognitive_scaffold import MetaConstraintLog, ConstraintEvent
from .agent_governance import Agent, GovernanceCoordinator
from .multimodal_ethics import AdaptiveEthics, EthicalRule
from .integration_loop import (
    CognitiveIntegrationLoop,
    ModelRegistry,
    ModelWrapper,
    CallPolicyEngine,
)

__all__ = [
    "HierarchicalMemory",
    "MetaConstraintLog",
    "ConstraintEvent",
    "Agent",
    "GovernanceCoordinator",
    "AdaptiveEthics",
    "EthicalRule",
    "CognitiveIntegrationLoop",
    "ModelRegistry",
    "ModelWrapper",
    "CallPolicyEngine",
]
