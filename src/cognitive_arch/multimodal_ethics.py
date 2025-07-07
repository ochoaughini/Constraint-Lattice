from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EthicalRule:
    name: str
    forbidden_terms: List[str]
    context: str | None = None


class AdaptiveEthics:
    """Very small ethical modulation engine.

    Checks a text for forbidden terms depending on context and can adapt rules
    dynamically at runtime.
    """

    def __init__(self, rules: List[EthicalRule] | None = None) -> None:
        self.rules = rules or []

    def check(self, text: str, context: str | None = None) -> List[str]:
        violations: List[str] = []
        for rule in self.rules:
            if rule.context and context and rule.context != context:
                continue
            for term in rule.forbidden_terms:
                if term.lower() in text.lower():
                    violations.append(rule.name)
                    break
        return violations

    def add_rule(self, rule: EthicalRule) -> None:
        self.rules.append(rule)
