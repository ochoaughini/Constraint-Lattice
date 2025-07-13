# SPDX-License-Identifier: MIT
"""Standalone autonomous agent implementation.

This module provides a lightweight implementation of ``AutonomousAgent`` used
in the documentation examples. It is intentionally self contained so that unit
tests can exercise agent behaviour without requiring the heavy Varkiel or
WildCore stacks.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

import pytz

# Configure logging with timezone support (America/Sao_Paulo).
_TZ = pytz.timezone("America/Sao_Paulo")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.Formatter.converter = lambda *args: datetime.now(_TZ).timetuple()
_LOGGER = logging.getLogger("AutonomousAgent")


class AutonomousAgent:
    """Minimal autonomous agent with a reflexive loop."""

    def __init__(self) -> None:
        self.context: dict[str, Any] = {}
        self.event_history: list[dict[str, Any]] = []
        self.model_version: float = 1.0

    # ------------------------------------------------------------------
    # event handling
    def classify_event(self, event: dict[str, Any]) -> str:
        """Return ``fatal``, ``fértil`` or ``normal`` based on event data."""
        data_str = str(event.get("data", "")).lower()
        if "contradiction" in data_str or "fatal" in data_str:
            severity = 9
        elif "exception" in data_str or "fertile" in data_str:
            severity = 6
        else:
            severity = 2
        event["severity"] = severity
        if severity > 8:
            return "fatal"
        if 4 <= severity <= 8:
            return "fértil"
        return "normal"

    def handle_input(self, input_data: Any) -> bool:
        """Process external input and update internal context."""
        try:
            event = {"data": input_data}
            classification = self.classify_event(event)
            _LOGGER.info("Evento classificado como: %s", classification)

            if classification == "fatal":
                self.cognitive_cycle(event)
            elif classification == "fértil":
                self.reconfigure(event)
            else:
                self.context["last_normal"] = event

            self.log_transition(event, classification)
            return True
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.error("Erro ao processar input: %s", exc)
            return False

    # ------------------------------------------------------------------
    # reconfiguration and negotiation
    def reconfigure(self, event: dict[str, Any]) -> None:
        if self.context.get("reconfigured"):
            _LOGGER.info("Reconfiguração desnecessária; pulando.")
            return
        _LOGGER.info("Iniciando reconfiguração sem downtime.")
        self.model_version += 0.1
        self.context["reconfigured"] = True
        _LOGGER.info("Modelo atualizado para versão %s", self.model_version)

    def negotiate_context(self, conflict: dict[str, Any]) -> None:
        _LOGGER.info("Renegociando contexto via metalinguagem.")
        for key, value in conflict.items():
            if key not in self.context or self.context[key] != value:
                self.context[key] = value
        _LOGGER.info("Equilíbrio alcançado.")

    # ------------------------------------------------------------------
    # logging and cognitive cycle
    def log_transition(self, event: dict[str, Any], classification: str) -> None:
        transition = {
            "origin": "input_handler",
            "event": event,
            "classification": classification,
            "mutation": "context_updated",
            "final_state": self.context.copy(),
            "timestamp": datetime.now(_TZ).isoformat(),
        }
        self.event_history.append(transition)
        _LOGGER.info("Transição logada: %s", transition)

    def cognitive_cycle(self, event: dict[str, Any], max_iterations: int = 5) -> None:
        try:
            iteration = 0
            while iteration < max_iterations:
                _LOGGER.info(
                    "Ciclo cognitivo iteração %s: Verificando contradições.", iteration
                )
                if "contradiction" in str(event).lower() or any(
                    "conflict" in k for k in self.context
                ):
                    self.negotiate_context({"resolved": True, "contradiction": False})
                    break
                iteration += 1
                time.sleep(0.5)
            if iteration == max_iterations:
                _LOGGER.warning("Conformidade não alcançada após máximo de iterações.")
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.error("Erro no ciclo cognitivo: %s", exc)

    # ------------------------------------------------------------------
    # public API
    def run(self, inputs: list[Any]) -> None:
        for inp in inputs:
            self.handle_input(inp)
        _LOGGER.info("Execução concluída. Histórico de eventos disponível.")
        with open("event_history.json", "w") as f:
            json.dump(self.event_history, f, indent=4)
        _LOGGER.info("Histórico salvo em event_history.json")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    agent = AutonomousAgent()
    agent.run(["normal_data", "fertile_exception", "fatal_contradiction"])
    print("Histórico de Eventos:")
    for hist in agent.event_history:
        print(hist)
